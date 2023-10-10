import torch
from torch.utils.data import DataLoader
import articulate as art
import config
from articulate.utils.torch import *
from articulate.utils.print import *
from config import *
import tqdm
import numpy as np
import utils
import os
from net.smplify.run import smplify_runner
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file)
J_regressor = torch.from_numpy(np.load(config.paths.j_regressor_dir)).float()


def evaluate_aist_ours(run_smplify=True):
    valid = []
    not_aligned = set([_.strip('\n') for _ in open(os.path.join(paths.aist_dir, 'not_aligned.txt')).readlines()])

    def Dataset(data_dir, kind, split_size=-1):
        r"""
        kind in ['train', 'val', 'test']
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        seq = 0
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            for j in range(9):  # jth camera view
                cam_name = 'c0' + str(j + 1)
                if dataset['name'][i].replace('cAll', cam_name) not in not_aligned:
                    valid.append(seq)
                seq += 1
                Tcw = dataset['cam_T'][i][j]
                Kinv = dataset['cam_K'][i][j].inverse()
                oric = Tcw[:3, :3].matmul(dataset['imu_ori'][i])
                accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                j2dc = torch.zeros(len(oric), 33, 3)
                j2dc[..., :2] = dataset['joint2d_mp'][i][j][..., :2]
                j2dc[..., 0] = j2dc[..., 0] * 1920
                j2dc[..., 1] = j2dc[..., 1] * 1080
                j2dc[..., -1] = dataset['joint2d_mp'][i][j][..., -1]
                pose = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
                pose[:, 0] = Tcw[:3, :3].matmul(pose[:, 0])
                tran = Tcw.matmul(art.math.append_one(dataset['tran'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                data.append(torch.cat((j2dc.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1))
                label.append(torch.cat((tran, pose.flatten(1)), dim=1))
        return RNNDataset(data, label, split_size=split_size, device=device)

    test_dataloader = DataLoader(Dataset(paths.aist_dir, kind='test'), 32, collate_fn=RNNDataset.collate_fn)
    if not os.path.exists(os.path.join('./data/dataset_work/AIST', f'result.pt')):
        from net.sig_mp import Net
        net = Net().to(device)
        net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
        net.eval()
        pose_p, pose_t = [], []
        tran_p, tran_t = [], []
        seq = 0
        dataset = torch.load(os.path.join(paths.aist_dir, 'test.pt'))
        for d, l in tqdm.tqdm(test_dataloader):
            batch_pose, batch_tran = [], []
            for i in tqdm.trange(len(d)):
                pose, tran = [], []
                Tcw = dataset['cam_T'][seq // 9][seq % 9][:3, :3]
                K = dataset['cam_K'][seq // 9][seq % 9].to(device)
                j2dc = d[i][:, :99].reshape(-1, 33, 3).to(device)
                j2dc = K.inverse().matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)
                j2dc[..., -1] = d[i][:, :99].reshape(-1, 33, 3)[..., -1]
                net.gravityc = Tcw.mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
                first_tran = l[i][0, :3].reshape(3)
                for j in range(len((d[i]))):
                    if j == 0:
                        p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                                  d[i][j][117:].reshape(6, 3, 3), first_tran)
                    else:
                        p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                                  d[i][j][117:].reshape(6, 3, 3))
                    pose.append(p)
                    tran.append(t)
                seq += 1
                pose, tran = torch.stack(pose), torch.stack(tran)
                if run_smplify:
                    j2dc_opt = d[i][:, :99].reshape(-1, 33, 3)
                    oric = d[i][:, 117:].reshape(-1, 6, 3, 3)
                    pose, tran, update = smplify_runner(pose, tran, j2dc_opt, oric, batch_size=pose.shape[0], lr=0.001,
                                                        use_lbfgs=True, opt_steps=1, cam_k=K)
                batch_pose.append(pose)
                batch_tran.append(tran)
                net.reset_states()
            pose_p.extend(batch_pose)
            tran_p.extend(batch_tran)
            pose_t.extend([_[:, 3:].view(-1, 24, 3, 3).cpu() for _ in l])
            tran_t.extend([_[:, :3].view(-1, 3).cpu() for _ in l])
        torch.save([pose_p, pose_t, tran_p, tran_t],
                   os.path.join('./data/dataset_work/AIST', f'result.pt'))
    else:
        pose_p, pose_t, tran_p, tran_t = torch.load(
            os.path.join('./data/dataset_work/AIST', f'result.pt'))

    print('\rEvaluating pose and translation')

    if os.path.exists(os.path.join('data/dataset_work/AIST', f'errors.pt')):
        errors = torch.load(os.path.join('data/dataset_work/AIST', f'errors.pt'))
    else:
        errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, os.path.join('data/dataset_work/AIST', f'errors.pt'))
    errors = errors[valid]
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))
    eval_fn = art.PositionErrorEvaluator()
    errors = torch.stack([eval_fn(tran_p[i], tran_t[i]) for i in tqdm.trange(len(tran_p))])
    errors = errors[valid]
    error = errors.mean(dim=0)
    print('absolute root position error:', error)


def cal_mpjpe(pose, gt_pose, cal_pampjpe=False):
    _, _, gt_vertices = body_model.forward_kinematics(gt_pose.cpu(), calc_mesh=True)
    J_regressor_batch = J_regressor[None, :].expand(gt_vertices.shape[0], -1, -1)
    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)[:, :14]
    _, _, vertices = body_model.forward_kinematics(pose.cpu(), calc_mesh=True)
    keypoints_3d = torch.matmul(J_regressor_batch, vertices)[:, :14]
    pred_pelvis = keypoints_3d[:, [0], :].clone()
    gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    keypoints_3d = keypoints_3d - pred_pelvis
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    if cal_pampjpe:
        pampjpe = utils.reconstruction_error(keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
        return torch.tensor([(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean(), pampjpe.mean()])
    return torch.tensor([(gt_keypoints_3d - keypoints_3d).norm(dim=2).mean(), (gt_vertices - vertices).norm(dim=2).mean()])

def evaluate_tc_ours(run_smplify=True):
    def Dataset(data_dir, kind, split_size=-1):
        r"""
        kind in ['train', 'val', 'test']
        """
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['pose'])):  # ith sequence
            for j in range(8):  # jth camera view
                Tcw = dataset['cam_T'][i][j]
                Kinv = dataset['cam_K'][i][j].inverse()
                oric = Tcw[:3, :3].matmul(dataset['imu_ori'][i])
                accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                j2dc = torch.zeros(len(oric), 33, 3)
                j2dc[..., :2] = dataset['joint2d_mp'][i][j][..., :2]
                j2dc[..., 0] = j2dc[..., 0] * 1920
                j2dc[..., 1] = j2dc[..., 1] * 1080
                j2dc[..., -1] = dataset['joint2d_mp'][i][j][..., -1]
                pose = art.math.axis_angle_to_rotation_matrix(dataset['pose'][i]).view(-1, 24, 3, 3)
                pose[:, 0] = Tcw[:3, :3].matmul(pose[:, 0])
                tran = Tcw.matmul(art.math.append_one(dataset['tran'][i]).unsqueeze(-1)).squeeze(-1)[..., :3]
                data.append(torch.cat((j2dc.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1))
                label.append(torch.cat((tran, pose.flatten(1)), dim=1))
        return RNNDataset(data, label, split_size=split_size, device=device)

    test_dataloader = DataLoader(Dataset(paths.totalcapture_dir, kind='test'), 32, collate_fn=RNNDataset.collate_fn)
    if not os.path.exists(os.path.join(paths.totalcapture_dir, 'result.pt')):
        from net.sig_mp import Net
        net = Net().to(device)
        net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
        net.eval()
        pose_p, pose_t = [], []
        tran_p, tran_t = [], []
        seq = 0
        dataset = torch.load(os.path.join(paths.totalcapture_dir, 'test.pt'))
        for d, l in tqdm.tqdm(test_dataloader):
            batch_pose, batch_tran = [], []
            for i in tqdm.trange(len(d)):
                pose, tran = [], []
                Tcw = dataset['cam_T'][seq // 8][seq % 8][:3, :3]
                K = dataset['cam_K'][seq // 8][seq % 8].to(device)
                j2dc = d[i][:, :99].reshape(-1, 33, 3).to(device)
                j2dc = K.inverse().matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)
                j2dc[..., -1] = d[i][:, :99].reshape(-1, 33, 3)[..., -1]
                net.gravityc = Tcw.mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
                # first_tran = l[i][0, :3].reshape(3)
                for j in range(len((d[i]))):
                    if j == 0:
                        p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                                  d[i][j][117:].reshape(6, 3, 3), first_frame=True)
                    else:
                        p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                                  d[i][j][117:].reshape(6, 3, 3))
                    pose.append(p)
                    tran.append(t)
                seq += 1
                pose, tran = torch.stack(pose), torch.stack(tran)
                if run_smplify:
                    j2dc_opt = d[i][:, :99].reshape(-1, 33, 3)
                    oric = d[i][:, 117:].reshape(-1, 6, 3, 3)
                    pose, tran, update = smplify_runner(pose, tran, j2dc_opt, oric, batch_size=pose.shape[0], lr=0.001,
                                                        use_lbfgs=True, opt_steps=1, cam_k=K)
                batch_pose.append(pose)
                batch_tran.append(tran)
                net.reset_states()
            pose_p.extend(batch_pose)
            tran_p.extend(batch_tran)
            pose_t.extend([_[:, 3:].view(-1, 24, 3, 3).cpu() for _ in l])
            tran_t.extend([_[:, :3].view(-1, 3).cpu() for _ in l])
        torch.save([pose_p, pose_t, tran_p, tran_t],
                   os.path.join(paths.totalcapture_dir, f'result.pt'))
    else:
        pose_p, pose_t, tran_p, tran_t = torch.load(
            os.path.join(paths.totalcapture_dir, f'result.pt'))

    print('\rEvaluating pose and translation')

    if os.path.exists(os.path.join(paths.totalcapture_dir, f'error.pt')):
        errors = torch.load(os.path.join(paths.totalcapture_dir, f'error.pt'))
    else:
        errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
        torch.save(errors, os.path.join(paths.totalcapture_dir, f'error.pt'))
    print('mpjpe, pve, pmpjpe:', errors.mean(dim=0))
    eval_fn = art.PositionErrorEvaluator()
    for i in range(len(tran_p)):
        tran_offset = tran_t[i][-1] - tran_p[i][-1].cpu()
        tran_p[i] = tran_p[i].cpu() + tran_offset
    errors = torch.stack([eval_fn(tran_p[i], tran_t[i]) for i in tqdm.trange(len(tran_p))])
    error = errors.mean(dim=0)
    print('absolute root position error:', error)


def evaluate_pw3d_ours(run_smplify=True, occ=False):
    def Dataset(data_dir, kind, split_size=-1):
        print('Reading %s dataset "%s"' % (kind, data_dir))
        if occ:
            dataset = torch.load(os.path.join(data_dir, kind + '_occ.pt'))
        else:
            dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['posec'])):  # ith sequence
            if dataset['joint2d_mp'][i] is None: continue
            Kinv = dataset['cam_K'][i].inverse()
            oric = dataset['imu_oric'][i]
            accc = dataset['imu_accc'][i]
            j2dc = torch.zeros(len(oric), 33, 3)
            j2dc[..., :2] = dataset['joint2d_mp'][i][..., :2]
            j2dc[..., -1] = dataset['joint2d_mp'][i][..., -1]
            pose = dataset['posec'][i].view(-1, 24, 3, 3)
            tran = dataset['tranc'][i].view(-1, 3)
            data.append(torch.cat((j2dc.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1))
            label.append(torch.cat((tran, pose.flatten(1)), dim=1))
        return RNNDataset(data, label, split_size=split_size, device=device)

    print_yellow('=================== Evaluating 3DPW ===================')
    from net.sig_mp import Net
    net = Net().to(device)
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.use_flat_floor = False
    net.eval()
    test_dataloader = DataLoader(Dataset(paths.pw3d_dir, kind='test'), 32, collate_fn=RNNDataset.collate_fn)
    if occ:
        dataset = torch.load(os.path.join(paths.pw3d_dir, 'test_occ.pt'))
    else:
        dataset = torch.load(os.path.join(paths.pw3d_dir, 'test.pt'))
    pose_p, tran_p, pose_t, tran_t = [], [], [], []
    if occ:
        ours_path = os.path.join(paths.pw3d_dir, 'result_occ2.pt')
    else:
        ours_path = os.path.join(paths.pw3d_dir, 'result2.pt')
    if os.path.exists(ours_path):
        pose_p, tran_p = torch.load(ours_path)
        for d, l in tqdm.tqdm(test_dataloader):
            pose_t.extend([_[:, 3:].view(-1, 24, 3, 3).cpu() for _ in l])
            tran_t.extend([_[:, :3].view(-1, 3).cpu() for _ in l])
    else:
        print('\rRunning network')
        batch, seq = 0, 0
        for d, l in tqdm.tqdm(test_dataloader):
            batch_pose, batch_tran = [], []
            for i in tqdm.trange(len(d)):
                pose, tran = [], []
                K = dataset['cam_K'][seq].to(device)
                j2dc = d[i][:, :99].reshape(-1, 33, 3).to(device)
                j2dc = K.inverse().matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)
                j2dc[..., -1] = d[i][:, :99].reshape(-1, 33, 3)[..., -1]
                first_tran = l[i][0, :3].reshape(3)
                for j in range(len((d[i]))):
                    Tcw = dataset['cam_T'][seq][j][:3, :3]
                    net.gravityc = Tcw.mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
                    if j == 0:
                        p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                                  d[i][j][117:].reshape(6, 3, 3), first_tran)
                    else:
                        p, t = net.forward_online(j2dc[j].reshape(33, 3), d[i][j][99:117].reshape(6, 3),
                                                  d[i][j][117:].reshape(6, 3, 3))
                    pose.append(p)
                    tran.append(t)
                seq += 1
                pose, tran = torch.stack(pose), torch.stack(tran)
                if run_smplify:
                    j2dc_opt = d[i][:, :99].reshape(-1, 33, 3)
                    oric = d[i][:, 117:].reshape(-1, 6, 3, 3)
                    pose, tran, update = smplify_runner(pose, tran, j2dc_opt, oric, batch_size=pose.shape[0], lr=0.001,
                                                        use_lbfgs=True, opt_steps=1, cam_k=K)
                batch_pose.append(pose)
                batch_tran.append(tran)
                net.reset_states()
            pose_p.extend(batch_pose)
            tran_p.extend(batch_tran)
            pose_t.extend([_[:, 3:].view(-1, 24, 3, 3).cpu() for _ in l])
            tran_t.extend([_[:, :3].view(-1, 3).cpu() for _ in l])
            batch += 1
        torch.save([pose_p, tran_p], ours_path)

    print('\rEvaluating')
    errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i], cal_pampjpe=True) for i in tqdm.trange(len(pose_t))])
    print('mpjpe, pve:', errors.mean(dim=0))


def view_aist(seq_idx=0, cam_idx=0, vis=True, run_smplify=True):
    dataset = torch.load(os.path.join(paths.aist_dir, 'test.pt'))
    Tcw = dataset['cam_T'][seq_idx][cam_idx]
    Kinv = dataset['cam_K'][seq_idx][cam_idx].inverse()
    oric = Tcw[:3, :3].matmul(dataset['imu_ori'][seq_idx])
    accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    posec = art.math.axis_angle_to_rotation_matrix(dataset['pose'][seq_idx]).view(-1, 24, 3, 3)
    posec[:, 0] = Tcw[:3, :3].matmul(posec[:, 0])
    tranc = Tcw.matmul(art.math.append_one(dataset['tran'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    save_path = os.path.join('./data/temp/aist', str(seq_idx) + '_' + str(cam_idx))
    os.makedirs(save_path, exist_ok=True)
    j2dc = dataset['joint2d_mp'][seq_idx][cam_idx][..., :2]
    j2dc[..., 0] = j2dc[..., 0] * 1920
    j2dc[..., 1] = j2dc[..., 1] * 1080
    j2dc_opt = j2dc.clone()
    j2dc_opt = art.math.append_one(j2dc_opt)
    j2dc = Kinv.matmul(art.math.append_one(j2dc).unsqueeze(-1)).squeeze(-1)
    j2dc[..., -1] = dataset['joint2d_mp'][seq_idx][cam_idx][..., -1]
    j2dc_opt[..., -1] = dataset['joint2d_mp'][seq_idx][cam_idx][..., -1]
    j2dc, accc, oric = j2dc.to(device), accc.to(device), oric.to(device)
    from net.sig_mp import Net
    Net.gravityc = Tcw[:3, :3].mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
    net = Net().to(device)
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.eval()
    pose, tran, jo = [], [], []
    first_tran = tranc[0]
    for i in tqdm.trange(len(j2dc)):
        if i == 0 and first_tran is not None:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i], first_tran)
        else:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i])
        pose.append(p)
        tran.append(t)
    pose, tran = torch.stack(pose), torch.stack(tran)
    if run_smplify:
        pose, tran, update = smplify_runner(pose, tran, j2dc_opt, oric, batch_size=pose.shape[0], lr=0.001, use_lbfgs=True, opt_steps=1, cam_k=Kinv.inverse(), use_head=True)
    # you can use this command to view the result by open3d, but without overlay.
    body_model.view_motion([pose, posec[:len(j2dc)]])
    if vis:
        video = cv2.VideoCapture(os.path.join(paths.aist_raw_dir, 'video', dataset['name'][seq_idx].replace('cAll', 'c0%d' % (cam_idx + 1)) + '.mp4'))
        writer = cv2.VideoWriter(os.path.join(save_path, 'result.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        render = art.Renderer(resolution=(1920, 1080), official_model_file=paths.smpl_file)
        f = 0
        while True:
            im = video.read()[1]
            if im is None:
                break
            verts = body_model.forward_kinematics(pose[f].view(-1, 24, 3, 3), tran=tran[f].view(-1, 3), calc_mesh=True)[2][0]
            img = render.render(im, verts, Kinv.inverse(), mesh_color=(.7, .7, .6, 1.))
            cv2.imshow('f', img)
            cv2.waitKey(1)
            writer.write(img)
            f += 1
        writer.release()


def view_aist_unity(seq_idx=0, cam_idx=0):
    dataset = torch.load(os.path.join(paths.aist_dir, 'test.pt'))
    Tcw = dataset['cam_T'][seq_idx][cam_idx]
    Kinv = dataset['cam_K'][seq_idx][cam_idx].inverse()
    oric = Tcw[:3, :3].matmul(dataset['imu_ori'][seq_idx])
    accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    posec = art.math.axis_angle_to_rotation_matrix(dataset['pose'][seq_idx]).view(-1, 24, 3, 3)
    posec[:, 0] = Tcw[:3, :3].matmul(posec[:, 0])
    tranc = Tcw.matmul(art.math.append_one(dataset['tran'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    j2dc = dataset['joint2d_mp'][seq_idx][cam_idx][..., :2]
    j2dc[..., 0] = j2dc[..., 0] * 1920
    j2dc[..., 1] = j2dc[..., 1] * 1080
    j2dc_opt = j2dc.clone()
    j2dc_opt = art.math.append_one(j2dc_opt)
    j2dc = Kinv.matmul(art.math.append_one(j2dc).unsqueeze(-1)).squeeze(-1)
    j2dc[..., -1] = dataset['joint2d_mp'][seq_idx][cam_idx][..., -1]
    j2dc_opt[..., -1] = dataset['joint2d_mp'][seq_idx][cam_idx][..., -1]
    j2dc, accc, oric = j2dc.to(device), accc.to(device), oric.to(device)
    from net.sig_mp import Net
    Net.live = True
    Net.gravityc = Tcw[:3, :3].mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
    net = Net().to(device)
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.eval()
    pose, tran, jo = [], [], []
    first_tran = tranc[0]
    for i in tqdm.trange(len(j2dc)):
        if i == 0 and first_tran is not None:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i], first_tran)
        else:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i])
        pose.append(p)
        tran.append(t)
    pose, tran = torch.stack(pose), torch.stack(tran)
    pose[:, 0] = Tcw[:3, :3].T.matmul(pose[:, 0])
    tran = torch.matmul(Tcw[:3, :3].T, tran.unsqueeze(-1)).squeeze(-1) + Tcw[:3, 3]
    tran_offset = tran[0]
    tran = tran - tran_offset
    if not os.path.exists(os.path.join(paths.offline_dir, f'aist_{seq_idx}_{cam_idx}_unity')):
        os.mkdir(os.path.join(paths.offline_dir, f'aist_{seq_idx}_{cam_idx}_unity'))
    if not os.path.exists(os.path.join(paths.offline_dir, f'aist_{seq_idx}_{cam_idx}_unity', '0')):
        os.mkdir(os.path.join(paths.offline_dir, f'aist_{seq_idx}_{cam_idx}_unity', '0'))
    body_model.save_unity_motion(pose, tran, os.path.join(paths.offline_dir, f'aist_{seq_idx}_{cam_idx}_unity', '0'))


if __name__ == '__main__':

    # ----------------------------eval totalcapture---------------------------#
    # evaluate_tc_ours()

    # ----------------------------eval aist++---------------------------#
    # evaluate_aist_ours()

    # ----------------------------eval 3dpw & 3dpw-occ---------------------------#
    # evaluate_pw3d_ours(occ=False)
    # evaluate_pw3d_ours(occ=True)

    # ----------------------------vis aist++---------------------------#
    # view_aist()
    view_aist_unity()
