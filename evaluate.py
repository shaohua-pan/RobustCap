import os
from functools import partial

import torch
from torch.utils.data import DataLoader
import articulate as art
import config
from articulate.utils.torch import *
from articulate.utils.print import *
from config import *
import tqdm
from net.sig_mp import Net
import cv2
import numpy as np
import utils
import os
import mediapipe as mp
import articulate.occlusion as occlusion
import random
from net.smplify.run import smplify_runner
import pickle
from articulate.filter import LowPassFilterRotation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file)
mp_detector = mp.solutions.pose.Pose(static_image_mode=True, enable_segmentation=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
J_regressor = torch.from_numpy(np.load(config.paths.j_regressor_dir)).float()


def get_bbox_scale(uv):
    r"""
    max(bbox width, bbox height)
    """
    u_max, u_min = uv[..., 0].max(dim=-1).values, uv[..., 0].min(dim=-1).values
    v_max, v_min = uv[..., 1].max(dim=-1).values, uv[..., 1].min(dim=-1).values
    return torch.max(u_max - u_min, v_max - v_min)


def run_mp(path, vis=True, occ=False, save=False, save_ori=True, fps=60, save_dir=None, mode='video', half=False):
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if save_dir:
            out = cv2.VideoWriter(os.path.join(save_dir, 'b.avi'), fourcc, fps, (1920, 1080), True)
        else:
            out = cv2.VideoWriter('b.avi', fourcc, fps, (1920, 1080), True)
    if save_ori:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if save_dir:
            out_ori = cv2.VideoWriter(os.path.join(save_dir, 'b_ori.avi'), fourcc, fps, (1920, 1080), True)
        else:
            out_ori = cv2.VideoWriter('b_ori.avi', fourcc, fps, (1920, 1080), True)
    if mode == 'video':
        cap = cv2.VideoCapture(path)
    uvs = []
    if occ:
        occluders = occlusion.load_occluders(config.paths.occ_dir)
        width_height = np.asarray([1920, 1080])
        im_scale_factor = min(width_height) / 256
        count = np.random.randint(4, 8)
        occs, centers = [], []
        for _ in range(count):
            occluder = random.choice(occluders)
            random_scale_factor = np.random.uniform(0.5, 1.0)
            scale_factor = random_scale_factor * im_scale_factor
            occluder = occlusion.resize_by_factor(occluder, scale_factor)
            center = np.random.uniform([600, 400], [1300, 960])
            occs.append(occluder)
            centers.append(center)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2) as pose:
        if mode == 'video':
            while cap.isOpened():
                success, image = cap.read()
                if image is None:
                    break
                if occ:
                    for _ in range(count):
                        occlusion.paste_over(im_src=occs[_], im_dst=image, center=centers[_])
                if save_ori:
                    if half:
                        image_h = image.copy()
                        image_h[:, :1060, 1], image_h[:, :1060, 2] = image_h[:, :1060, 0], image_h[:, :1060, 0]
                        out_ori.write(image_h[:, :960, :])
                    else:
                        out_ori.write(image)
                if half:
                    image[:, :1060, :] = 0
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if vis:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imshow('MediaPipe Pose', image)
                    cv2.waitKey(1)
                if save:
                    out.write(image)
                uv = []
                if results.pose_landmarks is None:
                    temp = torch.rand(33, 4)
                    temp[..., -1] = 0
                    uvs.append(temp)
                else:
                    for i in results.pose_landmarks.landmark:
                        uv.append([i.x, i.y, i.z, i.visibility])
                    uvs.append(torch.tensor(uv))
            cap.release()
        else:
            im_names = sorted(os.listdir(path))
            f = 0
            image = cv2.imread(os.path.join(path, im_names[0]))
            while True:
                if f == len(im_names):
                    break
                if occ:
                    for _ in range(count):
                        occlusion.paste_over(im_src=occs[_], im_dst=image, center=centers[_])
                if save_ori:
                    out_ori.write(image)
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if vis:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imshow('MediaPipe Pose', image)
                    cv2.waitKey(1)
                if save:
                    out.write(image)
                uv = []
                if results.pose_landmarks is None:
                    temp = torch.rand(33, 4)
                    temp[..., -1] = 0
                    uvs.append(temp)
                else:
                    for i in results.pose_landmarks.landmark:
                        uv.append([i.x, i.y, i.z, i.visibility])
                    uvs.append(torch.tensor(uv))
                image = cv2.imread(os.path.join(path, im_names[f]))
                f += 1
    if save:
        out.release()
    if save_ori:
        out_ori.release()
        if occ:
            torch.save(uvs, os.path.join(save_dir, 'uvs_occ.pt'))
        elif half:
            torch.save(uvs, os.path.join(save_dir, 'uvs_half.pt'))
        else:
            torch.save(uvs, os.path.join(save_dir, 'uvs.pt'))
    return uvs


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


def view_aist(seq_idx=177, cam_idx=0, occ=False, vis=True, run_smplify=True):
    # 1154, 1088 are no aligned
    dataset = torch.load(os.path.join(paths.aist_dir, 'test.pt'))
    Tcw = dataset['cam_T'][seq_idx][cam_idx]
    Kinv = dataset['cam_K'][seq_idx][cam_idx].inverse()
    oric = Tcw[:3, :3].matmul(dataset['imu_ori'][seq_idx])
    accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    j3dc = Tcw.matmul(art.math.append_one(dataset['joint3d'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    j3dc = j3dc[:, 1:] - j3dc[:, :1]
    posec = art.math.axis_angle_to_rotation_matrix(dataset['pose'][seq_idx]).view(-1, 24, 3, 3)
    posec[:, 0] = Tcw[:3, :3].matmul(posec[:, 0])
    tranc = Tcw.matmul(art.math.append_one(dataset['tran'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    save_path = os.path.join('./data/temp/aist', str(seq_idx) + '_' + str(cam_idx))
    os.makedirs(save_path, exist_ok=True)
    if occ:
        path = os.path.join(paths.aist_raw_dir, 'video', dataset['name'][seq_idx].replace('cAll', 'c0%d' % (cam_idx + 1)) + '.mp4')
        if not os.path.exists(os.path.join(save_path, 'uvs.pt')):
            uvs = run_mp(path, occ=True, save=True, save_ori=True, save_dir=save_path)
        else:
            uvs = torch.load(os.path.join(save_path, 'uvs.pt'))
        uvs = torch.stack(uvs)
        j2dc = uvs[..., :2]
    else:
        j2dc = dataset['joint2d_mp'][seq_idx][cam_idx][..., :2]
    j2dc[..., 0] = j2dc[..., 0] * 1920
    j2dc[..., 1] = j2dc[..., 1] * 1080
    j2dc_opt = j2dc.clone()
    j2dc_opt = art.math.append_one(j2dc_opt)
    j2dc = Kinv.matmul(art.math.append_one(j2dc).unsqueeze(-1)).squeeze(-1)
    if occ:
        j2dc[..., 2] = uvs[..., -1]
        j2dc_opt[..., 2] = uvs[..., -1]
    else:
        j2dc[..., -1] = dataset['joint2d_mp'][seq_idx][cam_idx][..., -1]
        j2dc_opt[..., -1] = dataset['joint2d_mp'][seq_idx][cam_idx][..., -1]
    j2dc, accc, oric = j2dc.to(device), accc.to(device), oric.to(device)
    from net.sig_mp import Net
    Net.gravityc = Tcw[:3, :3].mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
    net = Net().to(device)
    # net.use_reproj_opt = True
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
    error = cal_mpjpe(pose, posec[:len(j2dc)])
    print('mpjpe:', error)
    eval_fn = art.PositionErrorEvaluator()
    error = eval_fn(tran, tranc[:len(j2dc)])
    print('absolute root position error:', error)
    body_model.view_motion([pose, posec], [tran, tranc])
    if vis:
        if occ:
            video = cv2.VideoCapture(os.path.join(save_path, 'b_ori.avi'))
            writer = cv2.VideoWriter(os.path.join(save_path, 'result_occ.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        else:
            video = cv2.VideoCapture(os.path.join(paths.aist_raw_dir, 'video', dataset['name'][seq_idx].replace('cAll', 'c0%d' % (cam_idx + 1)) + '.mp4'))
            writer = cv2.VideoWriter(os.path.join(save_path, 'result.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        render = art.Renderer(resolution=(1920, 1080), official_model_file=paths.smpl_file)
        f = 0
        while True:
            im = video.read()[1]
            if im is None:
                break
            # im = np.zeros((1080, 1920, 3), dtype=np.uint8)
            # for i in range(33):
            #     cv2.circle(im, tuple(j2dc_opt[f][i][:2].cpu().numpy().astype(np.int32)), 3, (0, 255, 0), -1)
            # cv2.imwrite('b.jpg', im)
            verts = body_model.forward_kinematics(pose[f].view(-1, 24, 3, 3), tran=tran[f].view(-1, 3), calc_mesh=True)[2][0]
            img = render.render(im, verts, Kinv.inverse(), mesh_color=(.7, .7, .6, 1.))
            cv2.imshow('f', img)
            cv2.waitKey(1)
            writer.write(img)
            f += 1
        writer.release()


def view_tc(seq_idx=36, cam_idx=0, occ=False, vis=True, run_smplify=True, use_save_2d=True, half=False):
    dataset = torch.load(os.path.join(paths.totalcapture_dir, 'test.pt'))
    Tcw = dataset['cam_T'][seq_idx][cam_idx]
    Kinv = dataset['cam_K'][seq_idx][cam_idx].inverse()
    oric = Tcw[:3, :3].matmul(dataset['imu_ori'][seq_idx])
    accc = Tcw.matmul(art.math.append_zero(dataset['imu_acc'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    posec = art.math.axis_angle_to_rotation_matrix(dataset['pose'][seq_idx]).view(-1, 24, 3, 3)
    posec[:, 0] = Tcw[:3, :3].matmul(posec[:, 0])
    tranc = Tcw.matmul(art.math.append_one(dataset['tran'][seq_idx]).unsqueeze(-1)).squeeze(-1)[..., :3]
    _, j3dc = body_model.forward_kinematics(posec, tran=tranc)
    save_path = os.path.join('./data/temp/tc', str(seq_idx) + '_' + str(cam_idx))
    path = os.path.join(paths.totalcapture_raw_dir, 'video2/%s_cam%d.mp4' % (dataset['name'][seq_idx], cam_idx + 1))
    os.makedirs(save_path, exist_ok=True)
    if not use_save_2d:
        if occ:
            if not os.path.exists(os.path.join(save_path, 'uvs_occ.pt')):
                uvs = run_mp(path, occ=True, save=True, save_ori=True, save_dir=save_path)
            else:
                uvs = torch.load(os.path.join(save_path, 'uvs_occ.pt'))
        elif half:
            if not os.path.exists(os.path.join(save_path, 'uvs_half.pt')):
                uvs = run_mp(path, occ=False, save=True, save_ori=True, save_dir=save_path, half=True)
            else:
                uvs = torch.load(os.path.join(save_path, 'uvs_half.pt'))
        else:
            if not os.path.exists(os.path.join(save_path, 'uvs.pt')):
                uvs = run_mp(path, occ=False, save=True, save_ori=True, save_dir=save_path)
            else:
                uvs = torch.load(os.path.join(save_path, 'uvs.pt'))
        uvs = torch.stack(uvs)
        j2dc = torch.zeros((len(uvs), 33, 3))
        j2dc[..., :2] = uvs[..., :2]
        j2dc[..., 2] = uvs[..., -1]
        j2dc[..., 0] = j2dc[..., 0] * 1920
        j2dc[..., 1] = j2dc[..., 1] * 1080
    else:
        j2d_c = dataset['joint2d_mp'][seq_idx][cam_idx]
        j2d_c[..., 0] = j2d_c[..., 0] * 1920
        j2d_c[..., 1] = j2d_c[..., 1] * 1080
        j2dc = torch.zeros((len(j2d_c), 33, 3))
        j2dc[..., :2] = j2d_c[..., :2]
        j2dc[..., 2] = j2d_c[..., -1]

    j2dc_opt = j2dc.clone()
    j2dc[..., :2] = Kinv.matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
    j2dc, accc, oric = j2dc.to(device), accc.to(device), oric.to(device)
    from net.sig_mp import Net
    Net.gravityc = Tcw[:3, :3].mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
    net = Net().to(device)
    # net.tran_filter_num = 1.1
    # net.use_reproj_opt = True
    # net.use_flat_floor = False
    # net.conf_range = (-0.1, -0.01)
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.eval()
    pose, tran = [], []
    for i in tqdm.trange(0, len(accc)):
        if i == 0:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i], first_frame=True)
        else:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i])
        pose.append(p)
        tran.append(t)
    pose, tran = torch.stack(pose), torch.stack(tran)
    if run_smplify:
        pose, tran, update = smplify_runner(pose, tran, j2dc_opt, oric, batch_size=pose.shape[0], lr=0.001, use_lbfgs=True, opt_steps=1, cam_k=Kinv.inverse())
    error = cal_mpjpe(pose, posec[:len(pose)])
    print('mpjpe:', error)
    body_model.view_motion([pose, posec[:len(pose)]])
    # tran_offset = tranc[-1] - tran[-1]
    # tran = tran + tran_offset
    # eval_fn = art.PositionErrorEvaluator()
    # error = eval_fn(tran, tranc[:len(accc)])
    # print('mean position error:', error)
    # verts = body_model.forward_kinematics(pose[2320].view(-1, 24, 3, 3), tran=tran[2320].view(-1, 3), calc_mesh=True)[2][0]
    # K = Kinv.inverse()
    # render = art.Renderer(resolution=(1920, 1080), official_model_file=paths.smpl_file)
    # video = cv2.VideoCapture(path)
    # for _ in range(2326): video.read()
    # im = video.read()[1]
    # img = render.render(im, verts, K)
    # cv2.imshow('f', img)
    # cv2.waitKey(0)
    # body_model.view_motion([pose, posec[:len(j2dc)]], [tran, tranc[:len(j2dc)]])
    if vis:
        if occ:
            video = cv2.VideoCapture(os.path.join(save_path, 'b_ori.avi'))
            writer = cv2.VideoWriter(os.path.join(save_path, 'result_occ.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920, 1080))
        else:
            video = cv2.VideoCapture(path)
            if half:
                writer = cv2.VideoWriter(os.path.join(save_path, 'result_half.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920, 1080))
            else:
                writer = cv2.VideoWriter(os.path.join(save_path, 'result.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920, 1080))
        render = art.Renderer(resolution=(1920, 1080), official_model_file=paths.smpl_file)
        f = 0
        for i in range(100):
            video.read()
        while True:
            im = video.read()[1]
            # if im is None:
            #     break
            # if half:
            #     im[:, :1060, 1:] = im[:, :1060, :1]
            # if f >= len(pose):
            #     break
            if f < 1273:
                continue
            im[:, :960, 1:] = im[:, :960, :1]
            verts = body_model.forward_kinematics(pose[f].view(-1, 24, 3, 3), tran=tran[f].view(-1, 3), calc_mesh=True)[2][0]
            img = render.render(im, verts, Kinv.inverse(), mesh_color=(.7, .7, .6, 1.))
            cv2.imshow('f', img)
            cv2.waitKey(1)
            # writer.write(img)
            f += 1
        writer.release()


def view_pw3d(seq_idx=-1, occ=False, vis=True, use_save_2d=False, occ_data=False, f_smplify=True):
    if occ_data:
        dataset = torch.load(os.path.join(paths.pw3d_dir, 'test_occ.pt'))
    else:
        dataset = torch.load(os.path.join(paths.pw3d_dir, 'test.pt'))
    Tcw = dataset['cam_T'][seq_idx]
    Kinv = dataset['cam_K'][seq_idx].inverse()
    oric = dataset['imu_oric'][seq_idx]
    accc = dataset['imu_accc'][seq_idx]
    posec = dataset['posec'][seq_idx].view(-1, 24, 3, 3)
    tranc = dataset['tranc'][seq_idx].view(-1, 3)[::2]
    shape = dataset['shape'][seq_idx]
    name = dataset['name'][seq_idx]
    if not occ_data:
        save_path = os.path.join('./data/temp/3dpw', str(seq_idx))
    else:
        save_path = os.path.join('./data/temp/3dpw_occ', str(seq_idx))
    os.makedirs(save_path, exist_ok=True)
    if not use_save_2d:
        path = os.path.join(config.paths.pw3d_raw_dir, 'imageFiles', name)
        if occ:
            if not os.path.exists(os.path.join(save_path, 'uvs_occ.pt')):
                uvs = run_mp(path, occ=True, save=True, save_ori=True, save_dir=save_path,  mode='image', fps=30)
            else:
                uvs = torch.load(os.path.join(save_path, 'uvs_occ.pt'))
        else:
            if not os.path.exists(os.path.join(save_path, 'uvs.pt')):
                uvs = run_mp(path, occ=False, save=True, save_ori=True, save_dir=save_path, fps=30, mode='image')
            else:
                uvs = torch.load(os.path.join(save_path, 'uvs.pt'))
        uvs = torch.stack(uvs)
        j2dc = torch.zeros((len(uvs), 33, 3))
        j2dc[..., :2] = uvs[..., :2]
        j2dc[..., 2] = uvs[..., -1]
        joint_2d = []
        for index in range(len(j2dc)):
            if index == len(j2dc) - 1:
                joint_2d.append(j2dc[index])
                joint_2d.append(j2dc[index])
                continue
            joint_2d.append(j2dc[index])
            joint_2d.append((j2dc[index + 1] + j2dc[index]) / 2.0)
        j2dc = torch.stack(joint_2d)
        img_dir = os.path.join(config.paths.pw3d_raw_dir, 'imageFiles', name)
        im_names = sorted(os.listdir(img_dir))
        im = cv2.imread(os.path.join(img_dir, im_names[0]))
        j2dc[..., 0] = j2dc[..., 0] * im.shape[1]
        j2dc[..., 1] = j2dc[..., 1] * im.shape[0]
        j2dc = j2dc[:len(posec)]
    else:
        j2dc = dataset['joint2d_mp'][seq_idx]
    j2dc_opt = j2dc.clone()
    j2dc[..., :2] = Kinv.matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
    j2dc_clone = j2dc.clone()
    j2dc, accc, oric = j2dc.to(device), accc.to(device), oric.to(device)
    os.makedirs(save_path, exist_ok=True)
    from net.sig_mp import Net
    net = Net().to(device)
    net.use_flat_floor = False
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.eval()
    # whether to use this, sometimes is useful(3dpw)?
    net.use_reproj_opt = True
    # net.tran_filter_num = 1.2
    pose, tran, jo = [], [], []
    for i in tqdm.trange(len(j2dc)):
        Tcw = dataset['cam_T'][seq_idx][i][:3, :3]
        net.gravityc = Tcw.mm(torch.tensor([0, -1, 0.]).view(3, 1)).view(3)
        if i == 0:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i], first_tran=tranc[0])
        else:
            p, t = net.forward_online(j2dc[i], accc[i], oric[i])
        pose.append(p)
        tran.append(t)
    pose, tran = torch.stack(pose)[::2], torch.stack(tran)[::2]
    pose, tran, update = smplify_runner(pose, tran, j2dc_opt[::2], oric[::2], batch_size=pose.shape[0], lr=1e-3, use_lbfgs=True,
                                        opt_steps=1, cam_k=Kinv.inverse(), use_head=True)
    error = cal_mpjpe(pose, posec[:len(j2dc)][::2])
    print('mean position error, local rotation error, global rotation error:', error)
    eval_fn = art.PositionErrorEvaluator()
    error = eval_fn(tran, tranc[:len(tran)])
    print('mean position error:', error)
    if vis:
        img_dir = os.path.join(config.paths.pw3d_raw_dir, 'imageFiles', name)
        im_names = sorted(os.listdir(img_dir))
        f = 1
        im = cv2.imread(os.path.join(img_dir, im_names[0]))
        writer = cv2.VideoWriter(os.path.join(save_path, 'result.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (im.shape[1], im.shape[0]))
        render = art.Renderer(resolution=(im.shape[1], im.shape[0]), official_model_file=paths.smpl_file)
        while True:
            if f == len(im_names):
                break
            im = cv2.imread(os.path.join(img_dir, im_names[f]))
            verts = body_model.forward_kinematics(pose[f].view(-1, 24, 3, 3), tran=tran[f].view(-1, 3), calc_mesh=True, shape=shape)[2][0]
            img = render.render(im, verts, Kinv.inverse(), mesh_color=(.7, .7, .6, 1.))
            cv2.imshow('f', img)
            cv2.waitKey(1)
            writer.write(img)
            f += 1
        writer.release()


def evaluate_pw3d_online(run_smplify=True):
    def Dataset(data_dir, kind, split_size=-1):
        print('Reading %s dataset "%s"' % (kind, data_dir))
        dataset = torch.load(os.path.join(data_dir, kind + '.pt'))
        data, label = [], []
        for i in tqdm.trange(len(dataset['posec'])):  # ith sequence
            if dataset['joint2d_mp'][i] is None: continue
            Kinv = dataset['cam_K'][i].inverse()
            oric = dataset['imu_oric'][i]
            accc = dataset['imu_accc'][i]
            j2dc = torch.zeros(len(oric), 33, 3)
            j2dc[..., :2] = dataset['joint2d_mp'][i][..., :2]
            # j2dc = Kinv.matmul(art.math.append_one(j2dc[..., :2]).unsqueeze(-1)).squeeze(-1)
            j2dc[..., -1] = 1
            pose = dataset['posec'][i].view(-1, 24, 3, 3)
            tran = dataset['tranc'][i].view(-1, 3)
            data.append(torch.cat((j2dc.flatten(1), accc.flatten(1), oric.flatten(1)), dim=1))
            label.append(torch.cat((tran, pose.flatten(1)), dim=1))
        return RNNDataset(data, label, split_size=split_size, device=device)

    print_yellow('=================== Evaluating 3DPW ===================')
    from net.sig_mp import Net
    net = Net().to(device)
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.eval()
    test_dataloader = DataLoader(Dataset(paths.pw3d_dir, kind='test'), 32, collate_fn=RNNDataset.collate_fn)
    dataset = torch.load(os.path.join(paths.pw3d_dir, 'test.pt'))
    pose_p, tran_p, pose_t, tran_t = [], [], [], []
    if os.path.exists(os.path.join('./data/dataset_work/3DPW', f'result_0501.pt')):
        pose_p, tran_p = torch.load(os.path.join('./data/dataset_work/3DPW', f'result_0501.pt'))
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
        torch.save([pose_p, tran_p], os.path.join('./data/dataset_work/3DPW', f'result_0501.pt'))

    print('\rEvaluating')
    # eval_fn = art.MeanPerJointErrorEvaluator(paths.smpl_file, device=device)
    # errors = torch.stack([eval_fn(pose_p[i], pose_t[i]) for i in tqdm.trange(len(pose_p))])
    # error = errors.mean(dim=0)
    # print('mean position error, local rotation error, global rotation error:', error)
    # print(errors.cpu().numpy())
    #
    # eval_fn = art.MeanPerJointErrorEvaluator(paths.smpl_file, align_joint=-1)
    # errors = torch.stack([eval_fn(pose_p[i], pose_t[i]) for i in tqdm.trange(len(pose_p))])
    # error = errors.mean(dim=0)
    # print('mean position error, local rotation error, global rotation error(r/t/s):', error)
    #
    # eval_fn = art.MeshErrorEvaluator(paths.smpl_file, align_joint=-1)
    # errors = torch.stack([eval_fn(pose_p[i], pose_t[i]) for i in tqdm.trange(len(pose_p))])
    # error = errors.mean(dim=0)
    # print('mesh error:', error)
    errors = torch.stack([cal_mpjpe(pose_p[i], pose_t[i]) for i in tqdm.trange(len(pose_t))])
    print('mpjpe, pve:', errors.mean(dim=0))
    # for i in range(len(tran_p)):
    #     tran_offset = tran_t[i][-1] - tran_p[i][-1]
    #     tran_p[i] = tran_p[i] + tran_offset
    eval_fn = art.PositionErrorEvaluator()
    errors = torch.stack([eval_fn(tran_p[i], tran_t[i]) for i in tqdm.trange(len(tran_p))])
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



def view_offline(name=None, save=False):
    if name is None:
        return
    cap = cv2.VideoCapture(os.path.join(paths.offline_dir, name + '.mp4'))
    data = torch.load(os.path.join(paths.offline_dir, name + '.pt'))
    oris, accs, RCMs, uvds = data[0], data[1], data[2], data[3]
    index, index_c = -1, -1
    K = torch.tensor([[623.79949084, 0., 313.69863974], [0., 623.09646347, 236.76807598], [0., 0., 1.]])
    net = Net().to(device)
    net.gravityc = torch.matmul(RCMs[0], torch.tensor([0., -1, 0.]).unsqueeze(-1)).squeeze(-1)
    net.conf_range = (0.8, 0.9)
    net.tran_filter_num = 0.01
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.eval()
    render = art.Renderer(resolution=(480, 640), official_model_file=paths.smpl_file)
    poses, trans, j2dc_opts, oriss = [], [], [], []
    # cap.read()
    # cap.read()
    last_frame = None
    uv_pre = None
    uvs = []
    if save:
        out = cv2.VideoWriter(os.path.join(paths.offline_dir, name + '_out.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    out2 = cv2.VideoWriter(os.path.join(paths.offline_dir, name + '_oriv.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (640, 480))
    out3 = cv2.VideoWriter(os.path.join(paths.offline_dir, name + '_mp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (640, 480))
    with mp_pose.Pose(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.001,
            model_complexity=1) as mpd:
        while cap.isOpened() and index < len(data[0]) - 1:
            index += 1
            if index % 2 == 0:
                ret, frame = cap.read()
                last_frame = frame.copy()
            else:
                frame = last_frame
            if index < 100:
                continue
            # if index == 3600:
            #     ret, frame = cap.read()
            #     ret, frame = cap.read()
            # if index > 700:
            #     break
            if index % 2 == 0:
                out2.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = mpd.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if index == 3840:
                if results.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.imwrite('2dd.jpg', frame)
            uv = torch.rand(33, 3)
            uv[..., 2] = 0.
            empimage  = np.zeros((480, 640, 3), dtype=np.uint8)
            empimage.fill(255)
            if results.pose_landmarks is not None:
                uv = []
                for i in results.pose_landmarks.landmark:
                    uv.append([i.x * frame.shape[1], i.y * frame.shape[0], i.visibility])
                    if index == 3840:
                        cv2.circle(empimage, (int(i.x * frame.shape[1]), int(i.y * frame.shape[0])), 4, (0, 255, 0), -1)
                if index == 3840:
                    cv2.imwrite('2de.jpg', empimage)
                uv = torch.tensor(uv)
                j2dc_opt = uv.clone()
            uv[..., :2] = K.inverse().matmul(art.math.append_one(uv[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
            oric, accc, RCM = oris[index], accs[index], RCMs[index]
            # uv = uvds[index]
            if index == 100:
                pose, tran = net.forward_online(uv.to(device), accc.to(device), oric.to(device), first_frame=True)
            else:
                pose, tran = net.forward_online(uv.to(device), accc.to(device), oric.to(device))
            # root = RCM.T.matmul(pose[0])
            # pose[0] = root
            # tran = RCM.T.matmul(tran.unsqueeze(-1)).squeeze(-1)
            if index % 2 == 0:
                poses.append(pose)
                trans.append(tran)
                j2dc_opts.append(j2dc_opt)
                oriss.append(oric)
                out3.write(frame)
                uvs.append(uv)
                verts = body_model.forward_kinematics(pose.view(-1, 24, 3, 3), tran=tran.view(-1, 3), calc_mesh=True)[2][0]
                frame = render.render(frame, verts, K, mesh_color=(.7, .7, .6, 1.))
                cv2.imshow('f', frame)
                if save:
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    poses, trans, update = smplify_runner(torch.stack(poses), torch.stack(trans), torch.stack(j2dc_opts), torch.stack(oriss), batch_size=len(poses), lr=1e-1, use_lbfgs=True,
                                        opt_steps=1, cam_k=K)
    cap = cv2.VideoCapture(os.path.join(paths.offline_dir, name +'.mp4'))
    index, index2 = -1, -1
    while cap.isOpened() and index < len(data[0]) - 1:
        index += 1
        if index % 2 == 0:
            ret, frame = cap.read()
            last_frame = frame.copy()
        else:
            frame = last_frame
        if index < 100:
            continue
        if index % 2 == 0:
            index2 += 1
            if index == 3840:
                cv2.imwrite('input.jpg', frame)
                emptyimage = np.zeros((480, 640, 3), dtype=np.uint8)
                emptyimage.fill(255)
                joint = body_model.forward_kinematics(poses[index2].view(-1, 24, 3, 3), tran=trans[index2].view(-1, 3))[1][0]
                joint = joint / joint[..., -1:]
                joint = torch.matmul(K, joint.unsqueeze(-1)).squeeze(-1)[..., :2]
                for i in range(24):
                    cv2.circle(emptyimage, (int(joint[i, 0]), int(joint[i, 1])), 4, (0, 0, 0), -1)
                cv2.imwrite('2dp.jpg', emptyimage)
            verts = body_model.forward_kinematics(poses[index2].view(-1, 24, 3, 3), tran=trans[index2].view(-1, 3), calc_mesh=True)[2][0]
            frame = render.render(frame, verts, K, mesh_color=(.7, .7, .6, 1.))
            cv2.imshow('f', frame)
            if index == 3840:
                cv2.imwrite('motion.jpg', frame)
            if save:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if save:
        out.release()
    # torch.save(torch.stack(uvs), os.path.join(paths.offline_dir, name + '_ruvs.pt'))
    # trans = torch.stack(trans)
    # trans = trans - trans[0]
    # body_model.save_unity_motion(torch.stack(poses), trans, os.path.join(paths.offline_dir, name + '_0504'))
    # out2.release()
    # out3.release()
    # body_model.view_motion([torch.stack(poses)], [torch.stack(trans)], fps=30)

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


def live_offline(name='occ_cloth2', overlay=True, begin_idx=100, end_idx=-1, smplify_type=1, lr=1e-3):
    cap = cv2.VideoCapture(os.path.join(paths.live_dir, name + '.mp4'))
    data = torch.load(os.path.join(paths.live_dir, name + '.pt'))
    oris, accs, RCMs, uvds = data[0], data[1], data[2], data[3]
    index = -1
    K = torch.tensor([[623.79949084, 0., 313.69863974], [0., 623.09646347, 236.76807598], [0., 0., 1.]])
    net = Net().to(device)
    net.gravityc = torch.matmul(RCMs[0], torch.tensor([0., -1, 0.]).unsqueeze(-1)).squeeze(-1)
    if not overlay:
        net.conf_range = (0.85, 0.9)
        net.tran_filter_num = 0.01
    else:
        net.conf_range = (0.85, 0.9)
        # net.tran_filter_num = 0.01
        # net.use_reproj_opt = True
    net.load_state_dict(torch.load(os.path.join(paths.weight_dir, Net.name, 'best_weights.pt')))
    net.eval()
    render = art.Renderer(resolution=(480, 640), official_model_file=paths.smpl_file)
    poses, trans, j2dc_opts, oriss, imu_ori, imu_acc, gt = [], [], [], [], [], [], []
    last_frame = None
    uvs = []
    if overlay:
        if smplify_type == 1:
            out = cv2.VideoWriter(os.path.join(paths.live_dir, 'ours', name + '_out1.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        else:
            out = cv2.VideoWriter(os.path.join(paths.live_dir, 'ours', name + '_out2.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    out2 = cv2.VideoWriter(os.path.join(paths.live_dir, 'ours', name + '_oriv.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    out3 = cv2.VideoWriter(os.path.join(paths.live_dir, 'ours', name + '_mp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    filter = LowPassFilterRotation()
    with mp_pose.Pose(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.001,
            model_complexity=1) as mpd:
        while cap.isOpened() and index < len(oris) - 1:
            index += 1
            if index % 2 == 0:
                _, frame = cap.read()
                last_frame = frame.copy()
            else:
                frame = last_frame
            if index < begin_idx:
                continue
            if index > end_idx and end_idx > 0:
                break
            if index % 2 == 0:
                out2.write(frame)
            im = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = mpd.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            uv = torch.rand(33, 3)
            uv[..., 2] = 0.
            if results.pose_landmarks is not None:
                uv = []
                for i in results.pose_landmarks.landmark:
                    uv.append([i.x * frame.shape[1], i.y * frame.shape[0], i.visibility])
                uv = torch.tensor(uv)
                j2dc_opt = uv.clone()
            uv[..., :2] = K.inverse().matmul(art.math.append_one(uv[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
            oric, accc, RCM = oris[index], accs[index], RCMs[index]
            if index == begin_idx:
                pose, tran = net.forward_online(uv.to(device), accc.to(device), oric.to(device), first_frame=True)
            else:
                pose, tran = net.forward_online(uv.to(device), accc.to(device), oric.to(device))
            gt.append(art.math.rotation_matrix_to_axis_angle(pose).view(72))
            if not overlay:
                root = RCM.T.matmul(pose[0])
                pose[0] = root
                tran = RCM.T.matmul(tran.unsqueeze(-1)).squeeze(-1)
            if index % 2 == 0:
                if not overlay:
                    pose = filter(pose)
                poses.append(pose)
                trans.append(tran)
                j2dc_opts.append(j2dc_opt)
                oriss.append(oric)
                imu_ori.append(oris[index-1])
                imu_acc.append(accs[index-1])
                imu_ori.append(oric)
                imu_acc.append(accc)
                out3.write(frame)
                uvs.append(uv)
                if overlay and smplify_type != 2:
                    if smplify_type == 1:
                        pose, tran, update = smplify_runner(pose, tran, j2dc_opt, oric, batch_size=1, lr=lr, use_lbfgs=True, opt_steps=1, cam_k=K)
                    verts = body_model.forward_kinematics(pose.view(-1, 24, 3, 3), tran=tran.view(-1, 3), calc_mesh=True)[2][0]
                    im = render.render(im, verts, K, mesh_color=(.7, .7, .6, 1.))
                    cv2.imshow('f', im)
                    cv2.waitKey(1)
                    out.write(im)
    if overlay and smplify_type == 2:
        poses, trans, update = smplify_runner(torch.stack(poses), torch.stack(trans), torch.stack(j2dc_opts), torch.stack(oriss), batch_size=len(poses), lr=lr, use_lbfgs=True, opt_steps=1, cam_k=K)
        imu_ori = torch.stack(imu_ori)
        imu_acc = torch.stack(imu_acc)
        tc_seq = [2, 3, 0, 1, 4, 5]
        imu_ori = imu_ori[:, tc_seq]
        imu_acc = imu_acc[:, tc_seq]
        gt = torch.stack(gt)
        torch.save({'gt': gt, 'ori': imu_ori, 'acc': imu_acc}, os.path.join(paths.live_dir, 'tip', name + '.pt'))
        torch.save({'tran': trans}, os.path.join(paths.live_dir, 'tip', name + '_tran.pt'))
        cap = cv2.VideoCapture(os.path.join(paths.live_dir, name +'.mp4'))
        index, index2 = -1, -1
        while cap.isOpened() and index < len(data[0]) - 1:
            index += 1
            if index % 2 == 0:
                _, frame = cap.read()
                last_frame = frame.copy()
            else:
                frame = last_frame
            if index < begin_idx:
                continue
            if index > end_idx and end_idx > 0:
                break
            if index % 2 == 0:
                index2 += 1
                verts = body_model.forward_kinematics(poses[index2].view(-1, 24, 3, 3), tran=trans[index2].view(-1, 3), calc_mesh=True)[2][0]
                frame = render.render(frame, verts, K, mesh_color=(.7, .7, .6, 1.))
                cv2.imshow('f', frame)
                cv2.waitKey(1)
                out.write(frame)
        out.release()
    out2.release()
    out3.release()
    torch.save(torch.stack(uvs), os.path.join(paths.live_dir, 'ours', name + '_oriuvs.pt'))
    if not overlay:
        trans = torch.stack(trans)
        trans = trans - trans[0]
        trans = torch.zeros(trans.shape)
        body_model.save_unity_motion(torch.stack(poses), trans, os.path.join(paths.live_dir, name + 'sig_asia'))


if __name__ == '__main__':

    # ----------------------------eval totalcapture---------------------------#
    evaluate_tc_ours()

    # ----------------------------eval aist++---------------------------
    # evaluate_aist_ours()

    # ----------------------------eval 3dpw & 3dpw-occ---------------------------
    # evaluate_pw3d_ours(occ=False)
    # evaluate_pw3d_ours(occ=True)

    # ----------------------------eval qualitative result---------------------------
    # view_aist()
    # view_tc()
    # view_pw3d(seq_idx=12, occ=False, vis=True, use_save_2d=False, occ_data=True)
    # view_offline(name='occ_cloth')

    # ----------------------------live ---------------------------
    # live_offline(name='out-of-camera', overlay=True, begin_idx=100, end_idx=-1, smplify_type=2, lr=1e-1)
