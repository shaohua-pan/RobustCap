import articulate as art
import torch
import os

import config
from config import *
import pickle
import tqdm
import json
import glob
import numpy as np
import cv2

extkp_mask = torch.tensor(list(HUMBIBody33.extended_keypoints.values()))
# reference https://google.github.io/mediapipe/solutions/pose
body_model = art.ParametricModel(paths.smpl_file)
mp_mask = torch.tensor(config.mp_mask)
vi_mask = torch.tensor(config.vi_mask)
ji_mask = torch.tensor(config.ji_mask)


def _syn_acc(v, smooth_n=2):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def preprocess_aist(kinds=['test']):
    '''
    Preprocess AIST dataset and save.
    '''
    if kinds is None:
        kinds = ['val', 'test', 'train']
    print('Preprocessing AIST++ dataset')
    tran_offset = torch.tensor([-0.00217368, -0.240789175, 0.028583793])  # smpl root offset in mean shape
    for kind in kinds:
        names = [_.strip('\n') for _ in open(os.path.join(paths.aist_raw_dir, 'splits', 'pose_%s.txt' % kind)).readlines()]
        ignore_names = set([_.strip('\n') for _ in open(os.path.join(paths.aist_raw_dir, 'ignore_list.txt')).readlines()])
        ignore_names2 = set([_.strip('\n') for _ in open(os.path.join(paths.aist_raw_dir, 'ignore_minimalbody.txt')).readlines()])
        mapping = {_.split(' ')[0]: _.split(' ')[1].strip('\n') for _ in open(os.path.join(paths.aist_raw_dir, 'cameras', 'mapping.txt')).readlines()}
        n_succeed = 0
        preprocessed_data = {'name': [], 'pose': [], 'tran': [], 'joint2d': [], 'joint2d_minimalbody':[], 'joint2d_mp': [], 'joint2d_occ': [], 'joint3d': [], 'cam_K': [], 'cam_T': [], 'imu_ori': [], 'imu_acc': [], 'romp_pose': [], 'romp_tran': [], 'pare_pose': [], 'pare_tran': []}
        for name in tqdm.tqdm(names):
            smpl_data = pickle.load(open(os.path.join(paths.aist_raw_dir, 'motions', name + '.pkl'), 'rb'), encoding='latin1')
            kp_data = pickle.load(open(os.path.join(paths.aist_raw_dir, 'keypoints2d', name + '.pkl'), 'rb'), encoding='latin1')
            cam_data = json.load(open(os.path.join(paths.aist_raw_dir, 'cameras', mapping[name] + '.json')))
            if name in ignore_names: continue    # ignore by official
            if smpl_data['smpl_loss'] > 4 and kind != 'test': continue    # bad sequence
            if np.isnan(kp_data['keypoints2d']).max() and kind != 'test': continue    # nan 2d keypoints
            romp_pose, romp_tran, pare_pose, pare_tran, kp_mp, kp_minimalbody, kp_mp_occ = [None for _ in range(9)], [None for _ in range(9)], [None for _ in range(9)], [None for _ in range(9)], [None for _ in range(9)], [None for _ in range(9)], [None for _ in range(9)]
            for cid in range(9):
                kp_minimalbody_file = os.path.join(paths.aist_raw_dir, 'keypoints2d_minimalbody', name.replace('cAll', 'c0%d' % (cid + 1)) + '.pt')
                if not os.path.exists(kp_minimalbody_file) and kind == 'test':
                    assert False, 'Missing %s' % kp_minimalbody_file
                if os.path.exists(kp_minimalbody_file) and (
                        os.path.basename(kp_minimalbody_file)[:-3] not in ignore_names2 or kind == 'test'):
                    kp_minimalbody[cid] = torch.stack(torch.load(kp_minimalbody_file))[:, :, [1, 0, 2]]
                    n = kp_data['keypoints2d'].shape[1] - kp_minimalbody[cid].shape[0]
                    assert n >= 0 and torch.isnan(kp_minimalbody[cid]).sum() == 0
                    if n == 1:
                        kp_minimalbody[cid] = torch.cat((kp_minimalbody[cid], kp_minimalbody[cid][-1:]), dim=0)
                    elif n == 2:
                        mid = kp_data['keypoints2d'].shape[1] // 2
                        kp_minimalbody[cid] = torch.cat((kp_minimalbody[cid][:mid], kp_minimalbody[cid][mid - 1:], kp_minimalbody[cid][-1:]), dim=0)
                    elif n == 3:
                        mid1 = kp_data['keypoints2d'].shape[1] // 3
                        mid2 = mid1 * 2
                        kp_minimalbody[cid] = torch.cat((kp_minimalbody[cid][:mid1], kp_minimalbody[cid][mid1 - 1:mid2], kp_minimalbody[cid][mid2 - 1:], kp_minimalbody[cid][-1:]), dim=0)
                    elif n >= 4:
                        kp_minimalbody[cid] = None
                kp_mp_file = os.path.join(paths.aist_raw_dir, 'keypoints2d_mp', name.replace('cAll', 'c0%d' % (cid + 1)) + '.pt')
                if not os.path.exists(kp_mp_file) and kind == 'test':
                    assert False, 'Missing %s' % kp_mp_file
                if os.path.exists(kp_mp_file) and (os.path.basename(kp_mp_file)[:-3] not in ignore_names2 or kind == 'test'):
                    mp_data = torch.load(kp_mp_file)
                    if mp_data is None or len(mp_data) == 0:
                        print('error process' + kp_mp_file)
                        kp_mp[cid] = None
                    else:
                        for index in range(len(mp_data)):
                            if mp_data[index] is None:
                                mp_data[index] = torch.rand(33, 4)
                                mp_data[index][:, -1] = 0.
                        kp_mp[cid] = torch.stack(mp_data)
                        n = kp_data['keypoints2d'].shape[1] - kp_mp[cid].shape[0]
                        assert n >= 0 and torch.isnan(kp_mp[cid]).sum() == 0
                        if n == 1:
                            kp_mp[cid] = torch.cat((kp_mp[cid], kp_mp[cid][-1:]), dim=0)
                        elif n == 2:
                            mid = kp_data['keypoints2d'].shape[1] // 2
                            kp_mp[cid] = torch.cat((kp_mp[cid][:mid], kp_mp[cid][mid - 1:], kp_mp[cid][-1:]), dim=0)
                        elif n == 3:
                            mid1 = kp_data['keypoints2d'].shape[1] // 3
                            mid2 = mid1 * 2
                            kp_mp[cid] = torch.cat((kp_mp[cid][:mid1], kp_mp[cid][mid1 - 1:mid2], kp_mp[cid][mid2 - 1:], kp_mp[cid][-1:]), dim=0)
                        elif n >= 4:
                            kp_mp[cid] = None
                kp_mp_occ_file = os.path.join(paths.aist_raw_dir, 'keypoints2d_mp_occ', name.replace('cAll', 'c0%d' % (cid + 1)) + '.pt')
                if os.path.exists(kp_mp_occ_file) and (os.path.basename(kp_mp_occ_file)[:-3] not in ignore_names2 and kind != 'test'):
                    mp_data_occ = torch.load(kp_mp_occ_file)
                    if mp_data_occ is None or len(mp_data_occ) == 0:
                        print('error process' + kp_mp_occ_file)
                        kp_mp_occ[cid] = None
                    else:
                        for index in range(len(mp_data_occ)):
                            if mp_data_occ[index] is None:
                                mp_data_occ[index] = torch.rand(33, 4)
                                mp_data_occ[index][:, -1] = 0.
                        kp_mp_occ[cid] = torch.stack(mp_data_occ)
                        n = kp_data['keypoints2d'].shape[1] - kp_mp_occ[cid].shape[0]
                        assert n >= 0 and torch.isnan(kp_mp_occ[cid]).sum() == 0
                        if n == 1:
                            kp_mp_occ[cid] = torch.cat((kp_mp_occ[cid], kp_mp_occ[cid][-1:]), dim=0)
                        elif n == 2:
                            mid = kp_data['keypoints2d'].shape[1] // 2
                            kp_mp_occ[cid] = torch.cat((kp_mp_occ[cid][:mid], kp_mp_occ[cid][mid - 1:], kp_mp_occ[cid][-1:]), dim=0)
                        elif n == 3:
                            mid1 = kp_data['keypoints2d'].shape[1] // 3
                            mid2 = mid1 * 2
                            kp_mp_occ[cid] = torch.cat((kp_mp_occ[cid][:mid1], kp_mp_occ[cid][mid1 - 1:mid2], kp_mp_occ[cid][mid2 - 1:], kp_mp_occ[cid][-1:]), dim=0)
                        elif n >= 4:
                            kp_mp_occ[cid] = None

                # used for eval romp
                romp_file = os.path.join(r'F:\ShaohuaPan\dataset\AIST\all_video\romp_pts', name.replace('cAll', 'c0%d' % (cid + 1)) + '.pt')
                if os.path.exists(romp_file) and kind == 'test':
                    data = torch.load(romp_file)
                    ori = torch.stack([torch.from_numpy(data[i]['global_orient']) for i in range(len(data))]).squeeze(1)
                    rot = torch.stack([torch.from_numpy(data[i]['body_pose']) for i in range(len(data))]).squeeze(1)
                    tran = torch.stack([torch.from_numpy(data[i]['cam_trans']) for i in range(len(data))]).squeeze(1)
                    r = torch.randn(len(data), 72)
                    r[:, :3] = ori
                    r[:, 3:] = rot
                    r = art.math.axis_angle_to_rotation_matrix(r).view(-1, 24, 3, 3)
                    romp_pose[cid] = r
                    romp_tran[cid] = tran
                    n = kp_data['keypoints2d'].shape[1] - romp_pose[cid].shape[0]
                    assert n >= 0
                    if n == 1:
                        romp_pose[cid] = torch.cat((romp_pose[cid], romp_pose[cid][-1:]), dim=0)
                        romp_tran[cid] = torch.cat((romp_tran[cid], romp_tran[cid][-1:]), dim=0)
                    elif n == 2:
                        romp_pose[cid] = torch.cat((romp_pose[cid][:1], romp_pose[cid], romp_pose[cid][-1:]), dim=0)
                        romp_tran[cid] = torch.cat((romp_tran[cid][:1], romp_tran[cid], romp_tran[cid][-1:]), dim=0)
                    elif n == 3:
                        mid = kp_data['keypoints2d'].shape[1] // 2
                        romp_pose[cid] = torch.cat((romp_pose[cid][:1], romp_pose[cid][:mid], romp_pose[cid][mid - 1:], romp_pose[cid][-1:]), dim=0)
                        romp_tran[cid] = torch.cat((romp_tran[cid][:1], romp_tran[cid][:mid], romp_tran[cid][mid - 1:], romp_tran[cid][-1:]), dim=0)
                    elif n >= 4:
                        romp_pose[cid] = None
                        romp_tran[cid] = None
                else:
                    romp_pose[cid] = None
                    romp_tran[cid] = None
                pare_file = os.path.join(r'F:\ShaohuaPan\dataset\AIST\pare_pts', name.replace('cAll', 'c0%d' % (cid + 1)) + '.pt')
                if os.path.exists(pare_file) and kind == 'test':
                    data = torch.load(pare_file)
                    j = 0
                    pare_pose_list, pare_tran_list = [], []
                    tran_temp = torch.tensor([0., 0, 0])
                    for t in range(data[1]['frame_ids'][-1] + 1):
                        while data[1]['frame_ids'][j] < t:
                            j += 1
                        if data[1]['frame_ids'][j] != t:
                            p = torch.eye(3).repeat(24, 1).view(24, 3, 3)
                            p[0] = torch.matmul(torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float), p[0])
                            pare_pose_list.append(p)
                            pare_tran_list.append(tran_temp)
                        else:
                            pare_pose_list.append(data[1]['pose'][j])
                            tran_cur = torch.from_numpy(data[1]['pred_cam'][j])
                            tran_temp = torch.stack([tran_cur[1], tran_cur[2], 2 * 5000 / (224 * tran_cur[0] + 1e-9)], dim=-1)
                            pare_tran_list.append(tran_temp)
                            j += 1
                    pare_pose[cid] = torch.from_numpy(np.stack(pare_pose_list))
                    pare_tran[cid] = torch.stack(pare_tran_list)
                    n = kp_data['keypoints2d'].shape[1] - pare_pose[cid].shape[0]
                    assert n >= 0
                    if n == 1:
                        pare_pose[cid] = torch.cat((pare_pose[cid], pare_pose[cid][-1:]), dim=0)
                        pare_tran[cid] = torch.cat((pare_tran[cid], pare_tran[cid][-1:]), dim=0)
                    elif n == 2:
                        pare_pose[cid] = torch.cat((pare_pose[cid][:1], pare_pose[cid], pare_pose[cid][-1:]), dim=0)
                        pare_tran[cid] = torch.cat((pare_tran[cid][:1], pare_tran[cid], pare_tran[cid][-1:]), dim=0)
                    elif n == 3:
                        mid = kp_data['keypoints2d'].shape[1] // 2
                        pare_pose[cid] = torch.cat(
                            (pare_pose[cid][:1], pare_pose[cid][:mid], pare_pose[cid][mid - 1:], pare_pose[cid][-1:]),
                            dim=0)
                        pare_tran[cid] = torch.cat(
                            (pare_tran[cid][:1], pare_tran[cid][:mid], pare_tran[cid][mid - 1:], pare_tran[cid][-1:]),
                            dim=0)
                    elif n >= 4:
                        pare_pose[cid] = None
                        pare_tran[cid] = None
                else:
                    pare_pose[cid] = None
                    pare_tran[cid] = None
            scale = smpl_data['smpl_scaling'].item()
            pose = torch.from_numpy(smpl_data['smpl_poses']).float()
            tran = torch.from_numpy(smpl_data['smpl_trans']).float() / scale + tran_offset
            joint2d = torch.from_numpy(kp_data['keypoints2d']).float()
            cam_K = torch.stack([torch.tensor(d['matrix']) for d in cam_data])
            cam_R = torch.stack([art.math.axis_angle_to_rotation_matrix(torch.tensor(d['rotation']))[0] for d in cam_data])
            cam_t = torch.stack([torch.tensor(d['translation']) for d in cam_data]) / scale
            cam_T = art.math.transformation_matrix(cam_R, cam_t)

            p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
            gp, joint3d, vert = body_model.forward_kinematics(p, tran=tran, calc_mesh=True)
            # joint3d = torch.cat((joint3d[:, :-2], vert[:, extkp_mask]), dim=1)

            syn_3d = vert[:, mp_mask]
            imu_ori = gp[:, ji_mask].clone()
            imu_acc = _syn_acc(vert[:, vi_mask])

            assert joint2d.shape[1] == pose.shape[0] == tran.shape[0]
            assert joint2d.shape[0] == 9 and joint2d.shape[2] == 17
            assert all([cam_data[i]['name'] == 'c0%d' % (i + 1) and cam_data[i]['size'] == [1920, 1080] for i in range(9)])
            assert torch.isnan(pose).sum() == 0 and torch.isnan(tran).sum() == 0

            preprocessed_data['name'].append(name)
            preprocessed_data['pose'].append(pose)        # N, 72         local axis-angle
            preprocessed_data['tran'].append(tran)        # N, 3          in M
            preprocessed_data['joint2d'].append(joint2d)  # 9, N, 17, 3   in image
            preprocessed_data['joint3d'].append(joint3d)  # N, 33, 3      in M
            preprocessed_data['cam_K'].append(cam_K)      # 9, 3, 3       discard distortion
            preprocessed_data['cam_T'].append(cam_T)      # 9, 4, 4       Tcw
            preprocessed_data['imu_ori'].append(imu_ori)  # N, 6, 3, 3    in M
            preprocessed_data['imu_acc'].append(imu_acc)  # N, 6, 3       in M
            preprocessed_data['joint2d_mp'].append(kp_mp)
            preprocessed_data['joint2d_minimalbody'].append(kp_minimalbody)
            preprocessed_data['romp_pose'].append(romp_pose)
            preprocessed_data['romp_tran'].append(romp_tran)
            preprocessed_data['pare_pose'].append(pare_pose)
            preprocessed_data['pare_tran'].append(pare_tran)
            preprocessed_data['joint2d_occ'].append(kp_mp_occ)
            n_succeed += 1

        os.makedirs(paths.aist_dir, exist_ok=True)
        torch.save(preprocessed_data, os.path.join(paths.aist_dir, kind + '.pt'))
        print('Save %s.pt: succeed %d/%d' % (kind, n_succeed, len(names)))


def preprocess_amass():
    print('Preprocessing AMASS dataset')

    for kind in ['val', 'train']:
        data_pose, data_trans, data_beta, length = [], [], [], []
        for ds_name in getattr(amass_data, kind):
            print('\rReading', ds_name)
            for npz_fname in tqdm.tqdm(glob.glob(os.path.join(paths.amass_raw_dir, ds_name, ds_name, '*/*_poses.npz'))):
                try: cdata = np.load(npz_fname)
                except: continue

                framerate = int(cdata['mocap_framerate'])
                if framerate == 120: step = 2
                elif framerate == 60 or framerate == 59: step = 1
                else: continue

                data_pose.extend(cdata['poses'][::step].astype(np.float32))
                data_trans.extend(cdata['trans'][::step].astype(np.float32))
                data_beta.append(cdata['betas'][:10])
                length.append(cdata['poses'][::step].shape[0])

        assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py.'
        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
        pose[:, 23] = pose[:, 37]     # right hand
        pose = pose[:, :24].clone()   # only use body

        # align AMASS global fame with AIST
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print('Synthesizing IMU accelerations and orientations')
        b = 0
        preprocessed_data = {'pose': [], 'shape': [], 'tran': [], 'joint3d': [], 'imu_ori': [], 'imu_acc': [], 'sync_3d_mp': []}
        for i, l in tqdm.tqdm(list(enumerate(length))):
            if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
            p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            preprocessed_data['pose'].append(pose[b:b + l].clone())  # N, 24, 3
            preprocessed_data['tran'].append(tran[b:b + l].clone())  # N, 3
            preprocessed_data['shape'].append(shape[i].clone())  # 10
            # preprocessed_data['joint3d'].append(torch.cat((joint[:, :-2], vert[:, extkp_mask]), dim=1))  # N, 33, 3
            preprocessed_data['joint3d'].append(joint)
            preprocessed_data['sync_3d_mp'].append(vert[:, mp_mask]) # N, 33, 3
            preprocessed_data['imu_acc'].append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            preprocessed_data['imu_ori'].append(grot[:, ji_mask])  # N, 6, 3, 3
            b += l

        os.makedirs(paths.amass_dir, exist_ok=True)
        torch.save(preprocessed_data, os.path.join(paths.amass_dir, kind + '.pt'))
        print('Save %s.pt' % kind)


def preprocess_my_totalcapture_pre(debug=False):
    def _calculate_trans():
        f = open(os.path.join(base_path, 'Vicon_GroundTruth', subject, motion, 'gt_skel_gbl_pos.txt'))
        line = f.readline().split('\t')
        index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine', 'Hips']])
        pos = []
        while line:
            line = f.readline()
            pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
        pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
        return pos[:, 3]

    def _prepare_cam():
        f = open(os.path.join(base_path, 'calibration.cal'))
        line = f.readline().split('\t')
        while line:
            line = f.readline().split('\t')
            if line == '' or line[0] == '':
                break
            line = f.readline().split('\t')[0].split('\n')[0].split(' ')
            fx, fy, cx, cy = float(line[0]), float(line[1]), float(line[2]), float(line[3])
            cam_i = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            line = f.readline().split('\t')
            line = f.readline().split('\t')[0].split(' ')
            row1_1, row1_2, row1_3 = float(line[0]), float(line[1]), float(line[2])
            line = f.readline().split('\t')[0].split(' ')
            row2_1, row2_2, row2_3 = float(line[0]), float(line[1]), float(line[2])
            line = f.readline().split('\t')[0].split(' ')
            row3_1, row3_2, row3_3 = float(line[0]), float(line[1]), float(line[2])
            cam_r = torch.tensor([[row1_1, row1_2, row1_3], [row2_1, row2_2, row2_3], [row3_1, row3_2, row3_3]])
            line = f.readline().split('\t')[0].split(' ')
            t_1, t_2, t_3 = float(line[0]), float(line[1]), float(line[2])
            cam_t = torch.tensor([t_1, t_2, t_3])
            cams.append([cam_r, cam_t, cam_i])

    inches_to_meters = 0.0254
    base_path = paths.totalcapture_raw_dir
    poses, trans, oris, accs, cams, kp_2ds, kp_3ds, kp_3ds_pj, kp_mps = [], [], [], [], [], [], [], [], []
    _prepare_cam()
    for file in tqdm.tqdm(sorted(os.listdir(os.path.join(base_path, 'TotalCapture_60FPS_Original')))):
        data = pickle.load(open(os.path.join(base_path, 'TotalCapture_60FPS_Original', file), 'rb'),
                           encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = art.math.axis_angle_to_rotation_matrix(torch.from_numpy(data['gt']).float()).reshape(-1, 24, 3, 3)
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]
        pose[:, 0] = torch.matmul(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32),
                                  pose[:, 0])
        poses.append(pose)
        oris.append(torch.matmul(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32), ori))
        accs.append(torch.matmul(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32),
                                 acc.unsqueeze(-1)).squeeze(-1))
        subject = file.split('_')[0].upper()
        motion = file.split('.')[0].split('_')[1]
        kp_2d, kp_mp = [], []
        for i in range(8):
            kp_2d_cam = torch.load(
                os.path.join(base_path, 'kp2d', subject.lower() + '_' + motion + '_cam' + str(i + 1) + '.pt'))
            kp_2d.append(torch.from_numpy(np.asarray(kp_2d_cam)))
            mp_data = torch.load(os.path.join(base_path, 'kp2d_mp', subject.lower() + '_' + motion + '_cam' + str(i + 1) + '.pt'))
            for index in range(len(mp_data)):
                if mp_data[index] is None or len(mp_data[index]) == 0:
                    mp_data[index] = torch.rand(33, 4)
                    mp_data[index][:, -1] = 0.
            kp_mp.append(torch.stack(mp_data))
        tran = _calculate_trans()
        if acc.shape[0] < tran.shape[0]:
            tran = tran[:acc.shape[0]]
        assert tran.shape[0] == acc.shape[0] and acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        tran[:, 0] = tran[:, 0] - 0.03
        tran[:, 1] = tran[:, 1] + (1 / (10 + tran[:, 2]))
        trans.append(tran)
        _, kp_3d, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        # kp_3ds.append(torch.cat((kp_3d[:, :-2], vert[:, extkp_mask]), dim=1))
        kp_3ds.append(kp_3d)
        kp_2ds.append(kp_2d)
        kp_mps.append(kp_mp)
    os.makedirs(paths.totalcapture_raw_dir, exist_ok=True)
    print('saving')
    torch.save(
        {'pose': poses, 'tran': trans, 'ori': oris, 'acc': accs, 'cam': cams, 'kp_2d': kp_2ds, 'kp_3d': kp_3ds, 'kp_3ds_pj': kp_3ds_pj, 'kp_mp': kp_mps},
        os.path.join(paths.totalcapture_raw_dir, f'total_capture_data.pt'))

def preprocess_my_totalcapture():
    print('Preprocessing TotalCapturee dataset')
    data = torch.load(os.path.join(paths.totalcapture_raw_dir, 'total_capture_data.pt'))
    videos = []
    for file in tqdm.tqdm(sorted(os.listdir(os.path.join(paths.totalcapture_raw_dir, 'TotalCapture_60FPS_Original')))):
        subject = file.split('_')[0].upper()
        motion = file.split('.')[0].split('_')[1]
        video = set([_[:-9] for _ in os.listdir(os.path.join(paths.totalcapture_raw_dir, 'video', subject, motion))])
        videos.extend(list(video))
    videos = sorted(list(videos))
    newdata = {'name': [], 'pose': [], 'tran': [], 'joint2d_minimalbody': [], 'joint2d_mp':[], 'joint3d': [], 'cam_K': [], 'cam_T': [], 'imu_ori': [], 'imu_acc': []}

    Rs, ts, Ks = [], [], []
    for j in range(8):
        R, t, K = data['cam'][j]
        Rs.append(R)
        ts.append(t)
        Ks.append(K)
    cam_T = art.math.transformation_matrix(torch.stack(Rs), torch.stack(ts))
    cam_K = torch.stack(Ks)

    for i in tqdm.trange(45):
        if i == 2 or i == 12 or i == 42: continue    # video-motion not aligned
        pose = data['pose'][i]
        tran = data['tran'][i]
        real_imu_ori = data['ori'][i]
        real_imu_acc = data['acc'][i]
        real_joint2d = data['kp_2d'][i]
        real_joint3d = data['kp_3d'][i]
        real_jointmp = []
        for j in range(len(data['kp_mp'][i])):
            real_jointmp.append(data['kp_mp'][i][j][:len(pose)])

        grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        syn_imu_ori = grot[:, ji_mask]
        syn_joint3d = torch.cat((joint[:, :-2], vert[:, extkp_mask]), dim=1)

        newdata['name'].append(videos[i])
        newdata['pose'].append(art.math.rotation_matrix_to_axis_angle(pose).view(-1, 24, 3))
        newdata['tran'].append(tran)
        newdata['joint2d_minimalbody'].append(torch.stack(real_joint2d)[..., [1, 0, 2]])
        newdata['joint2d_mp'].append(torch.stack(real_jointmp))
        newdata['cam_K'].append(cam_K)
        newdata['cam_T'].append(cam_T)
        newdata['imu_ori'].append(real_imu_ori)
        newdata['imu_acc'].append(real_imu_acc)
        newdata['joint3d'].append(real_joint3d)

        assert pose.shape[0] == tran.shape[0] == real_imu_ori.shape[0] == real_imu_acc.shape[0]
        assert art.math.radian_to_degree(art.math.angle_between(real_imu_ori, syn_imu_ori).mean()) < 17
        assert (real_joint3d[:, :22] - syn_joint3d[:, :22]).sum() < 0.01

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save(newdata, os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Save as test.pt')

def preprocess_3dpw():
    print('Preprocessing 3DPW dataset')
    newdata = {'name': [], 'posec': [], 'tranc': [], 'joint2d_mp': [], 'joint3d': [], 'cam_K': [], 'cam_T': [], 'imu_oric': [], 'imu_accc': [],'shape':[]}
    sequences = [x.split('.')[0] for x in os.listdir(os.path.join(paths.pw3d_raw_dir, 'sequenceFiles', 'test'))]
    for name in tqdm.tqdm(sequences):
        data = pickle.load(open(os.path.join(paths.pw3d_raw_dir, 'sequenceFiles', 'test', name + '.pkl'), 'rb'), encoding='latin1')
        num_people = len(data['poses'])
        for p_id in range(num_people):
            pose = torch.from_numpy(data['poses_60Hz'][p_id]).float()
            shape = torch.from_numpy(data['betas'][p_id][:10]).float()
            cam_pose = torch.from_numpy(np.repeat(data['cam_poses'], 2, axis=0)).float()
            trans = torch.from_numpy(data['trans_60Hz'][p_id]).float()[:len(cam_pose)]
            cam_intrinsics = torch.from_numpy(data['cam_intrinsics']).float()
            posec = art.math.axis_angle_to_rotation_matrix(pose.reshape(-1, 24, 3)).view(-1, 24, 3, 3)[:len(cam_pose)]
            posec[:, 0] = torch.matmul(cam_pose[:, :3, :3], posec[:, 0])
            tranc = cam_pose.matmul(art.math.append_one(trans).unsqueeze(-1)).squeeze(-1)[..., :3]
            grot, joint, vert = body_model.forward_kinematics(posec, shape=shape, tran=tranc, calc_mesh=True)
            oric = grot[:, ji_mask]
            accc = _syn_acc(vert[:, vi_mask])
            joint_2d = []
            mp_data = torch.load(os.path.join(paths.pw3d_raw_dir, 'kp2d_mp', name + '_' + str(p_id) + '.pt'))
            for index in range(len(mp_data)):
                if mp_data[index] is None or len(mp_data[index]) == 0:
                    mp_data[index] = torch.rand(33, 3)
                    mp_data[index][:, 2] = 0.
            for index in range(len(mp_data)):
                if index == len(mp_data) - 1:
                    joint_2d.append(mp_data[index])
                    joint_2d.append(mp_data[index])
                    continue
                joint_2d.append(mp_data[index])
                joint_2d.append((mp_data[index + 1] + mp_data[index]) / 2.0)
            newdata['name'].append(name)
            newdata['posec'].append(posec)
            newdata['tranc'].append(tranc)
            newdata['joint2d_mp'].append(torch.stack(joint_2d).float())
            newdata['joint3d'].append(joint)
            newdata['cam_K'].append(cam_intrinsics)
            newdata['cam_T'].append(cam_pose)
            newdata['imu_oric'].append(oric)
            newdata['imu_accc'].append(accc)
            newdata['shape'].append(shape)
            assert posec.shape[0] == tranc.shape[0] == oric.shape[0] == accc.shape[0] == len(joint_2d)
    os.makedirs(paths.pw3d_dir, exist_ok=True)
    torch.save(newdata, os.path.join(paths.pw3d_dir, 'test.pt'))
    print('Save as test.pt')


def preprocess_aist_pre():
    print('filter some not aligned data')
    kind = 'test'
    tran_offset = torch.tensor([-0.00217368, -0.240789175, 0.028583793])  # smpl root offset in mean shape
    names = [_.strip('\n') for _ in open(os.path.join(paths.aist_raw_dir, 'splits', 'pose_%s.txt' % kind)).readlines()]
    ignore_names = set([_.strip('\n') for _ in open(os.path.join(paths.aist_raw_dir, 'ignore_list.txt')).readlines()])
    not_aligned = open(os.path.join(paths.aist_raw_dir, 'not_aligned.txt'), 'w')
    mapping = {_.split(' ')[0]: _.split(' ')[1].strip('\n') for _ in open(os.path.join(paths.aist_raw_dir, 'cameras', 'mapping.txt')).readlines()}
    for name in tqdm.tqdm(names):
        smpl_data = pickle.load(open(os.path.join(paths.aist_raw_dir, 'motions', name + '.pkl'), 'rb'), encoding='latin1')
        kp_data = pickle.load(open(os.path.join(paths.aist_raw_dir, 'keypoints2d', name + '.pkl'), 'rb'), encoding='latin1')
        cam_data = json.load(open(os.path.join(paths.aist_raw_dir, 'cameras', mapping[name] + '.json')))
        if name in ignore_names: continue    # ignore by official
        cam_K = torch.stack([torch.tensor(d['matrix']) for d in cam_data])
        cam_R = torch.stack([art.math.axis_angle_to_rotation_matrix(torch.tensor(d['rotation']))[0] for d in cam_data])
        scale = smpl_data['smpl_scaling'].item()
        cam_t = torch.stack([torch.tensor(d['translation']) for d in cam_data]) / scale
        cam_T = art.math.transformation_matrix(cam_R, cam_t)
        for cid in range(9):
            kp_mp_file = os.path.join(paths.aist_raw_dir, 'keypoints2d_mp', name.replace('cAll', 'c0%d' % (cid + 1)) + '.pt')
            mp_data = torch.load(kp_mp_file)
            for index in range(len(mp_data)):
                if mp_data[index] is None or len(mp_data[index]) == 0:
                    mp_data[index] = torch.rand(33, 4)
                    mp_data[index][:, -1] = 0.
            kp_mp = torch.stack(mp_data)
            n = kp_data['keypoints2d'].shape[1] - kp_mp.shape[0]
            if n == 1:
                kp_mp = torch.cat((kp_mp, kp_mp[-1:]), dim=0)
            elif n == 2:
                mid = kp_data['keypoints2d'].shape[1] // 2
                kp_mp = torch.cat((kp_mp[:mid], kp_mp[mid - 1:], kp_mp[-1:]), dim=0)
            elif n == 3:
                mid1 = kp_data['keypoints2d'].shape[1] // 3
                mid2 = mid1 * 2
                kp_mp = torch.cat((kp_mp[:mid1], kp_mp[mid1 - 1:mid2], kp_mp[mid2 - 1:], kp_mp[-1:]), dim=0)
            kp_mp[..., :2] = kp_mp[..., :2] * torch.tensor([1920, 1080])
            pose = torch.from_numpy(smpl_data['smpl_poses']).float()
            tran = torch.from_numpy(smpl_data['smpl_trans']).float() / scale + tran_offset
            p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
            p[:, 0] = torch.matmul(cam_R[cid], p[:, 0])
            tran = torch.matmul(cam_T[cid], art.math.append_one(tran).unsqueeze(-1)).squeeze(-1)[..., :3]
            gp, joint3d, vert = body_model.forward_kinematics(p, tran=tran, calc_mesh=True)
            syn_3d = vert[:, mp_mask]
            syn_2d = syn_3d / syn_3d[..., -1:]
            syn_2d = torch.matmul(cam_K[cid], syn_2d.unsqueeze(-1)).squeeze(-1)[..., :2]
            d = ((kp_mp[..., :2] - syn_2d).norm(dim=-1)).mean()
            if d > 25:
                cam_id, cam_name = cid, 'c0' + str(cid + 1)
                # video_name = os.path.join(paths.aist_raw_dir, 'video', name.replace('cAll', cam_name) + '.mp4')
                # video = cv2.VideoCapture(video_name)
                # for i in range(syn_3d.shape[0]):
                #     succeed, im = video.read()
                #     if not succeed: break
                #     for kp in syn_2d[i]:
                #         im = cv2.circle(im, ((kp[0]).int().item(), (kp[1]).int().item()), radius=5, color=[0, 0, 255],
                #                         thickness=-1)  # red
                #     cv2.imshow('im', im)
                #     cv2.waitKey(1)
                print(name, cam_name, d)
                not_aligned.write(name.replace('cAll', cam_name) + '\n')
    not_aligned.close()

def preprocess_3dpwocc():
    print('Preprocessing 3DPW dataset')
    newdata = {'name': [], 'posec': [], 'tranc': [], 'joint2d_mp': [], 'joint3d': [], 'cam_K': [], 'cam_T': [], 'imu_oric': [], 'imu_accc': [], 'shape':[]}
    sequences = set([x.split('_')[0] + '_' + x.split('_')[1] + '_' +  x.split('_')[2] for x in os.listdir(os.path.join(paths.pw3d_raw_dir, 'kp2d_occ_mp'))])
    for name in tqdm.tqdm(sequences):
        data = pickle.load(open(os.path.join(paths.pw3d_raw_dir, 'sequenceFiles', 'all', name + '.pkl'), 'rb'), encoding='latin1')
        num_people = len(data['poses'])
        for p_id in range(num_people):
            pose = torch.from_numpy(data['poses_60Hz'][p_id]).float()
            shape = torch.from_numpy(data['betas'][p_id][:10]).float()
            cam_pose = torch.from_numpy(np.repeat(data['cam_poses'], 2, axis=0)).float()
            trans = torch.from_numpy(data['trans_60Hz'][p_id]).float()[:len(cam_pose)]
            cam_intrinsics = torch.from_numpy(data['cam_intrinsics']).float()
            posec = art.math.axis_angle_to_rotation_matrix(pose.reshape(-1, 24, 3)).view(-1, 24, 3, 3)[:len(cam_pose)]
            cam_pose = cam_pose[:len(posec)]
            posec[:, 0] = torch.matmul(cam_pose[:, :3, :3], posec[:, 0])
            tranc = cam_pose.matmul(art.math.append_one(trans).unsqueeze(-1)).squeeze(-1)[..., :3]
            grot, joint, vert = body_model.forward_kinematics(posec, shape=shape, tran=tranc, calc_mesh=True)
            oric = grot[:, ji_mask]
            accc = _syn_acc(vert[:, vi_mask])
            joint_2d = []
            mp_data = torch.load(os.path.join(paths.pw3d_raw_dir, 'kp2d_occ_mp', name + '_' + str(p_id) + '.pt'))
            for index in range(len(mp_data)):
                if mp_data[index] is None or len(mp_data[index]) == 0:
                    mp_data[index] = torch.rand(33, 3)
                    mp_data[index][:, 2] = 0.
            for index in range(len(mp_data)):
                if index == len(mp_data) - 1:
                    joint_2d.append(mp_data[index])
                    joint_2d.append(mp_data[index])
                    continue
                joint_2d.append(mp_data[index])
                joint_2d.append((mp_data[index + 1] + mp_data[index]) / 2.0)
            newdata['name'].append(name)
            newdata['posec'].append(posec)
            newdata['tranc'].append(tranc)
            newdata['joint2d_mp'].append(torch.stack(joint_2d)[:len(posec)].float())
            newdata['joint3d'].append(joint)
            newdata['cam_K'].append(cam_intrinsics)
            newdata['cam_T'].append(cam_pose)
            newdata['imu_oric'].append(oric)
            newdata['imu_accc'].append(accc)
            newdata['shape'].append(shape)
            assert posec.shape[0] == tranc.shape[0] == oric.shape[0] == accc.shape[0]
    os.makedirs(paths.pw3d_dir, exist_ok=True)
    torch.save(newdata, os.path.join(paths.pw3d_dir, 'test_occ.pt'))
    print('Save as test.pt')

if __name__ == '__main__':
    # preprocess_aist_pre()
    preprocess_aist()
    # preprocess_amass()
    # preprocess_my_totalcapture_pre(debug=True)
    # preprocess_my_totalcapture()
    # preprocess_3dpw()
    # preprocess_3dpwocc()
