# This script is the extended version of https://github.com/nkolot/SPIN/blob/master/smplify/smplify.py to deal with
# sequences inputs.

import os

import smplx
import torch

import config
# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior
import articulate as art
from config import paths
from smplx import SMPL
from .losses import temporal_ori_tran_fitting_loss, temporal_body_fitting_loss
from utils import sync_mp3d_from_smpl
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file, device=device)
tran_offset = torch.tensor([-0.00217368, -0.240789175, 0.028583793]).to(device)  # smpl root offset in mean shape
joint_mask = config.ji_mask

def batch_rodrigues(
    rot_vecs,
    epsilon=1e-8,
):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

class TemporalSMPLify():
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 cam_k,
                 imu_ori,
                 step_size=1.0,
                 num_iters=1,
                 use_lbfgs=True,
                 device=device,
                 batch_size=1,
                 max_iter=20,
                 shape=None,
                 use_head=False):

        # Store options
        self.device = device
        self.step_size = step_size
        self.max_iter = max_iter
        self.num_iters = num_iters
        self.cam_k = cam_k.detach().clone()
        self.imu_ori = imu_ori.detach().clone()
        self.batch_size = batch_size
        self.shape = shape
        if shape is not None:
            self.shape = shape.to(device).detach()

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='data/dataset_work', num_gaussians=8, dtype=torch.float32).to(device)
        self.use_lbfgs = use_lbfgs
        # self.smpl = SMPL(paths.smpl_file, batch_size=batch_size, gender='MALE').to(self.device)
        self.ign_mp_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32]
        if use_head:
            self.ign_mp_joints = [31, 32]


    def __call__(self, init_pose, init_tran, keypoints_2d):

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        # body_pose = init_pose[:, 3:].detach().clone()
        # global_orient = init_pose[:, :3].detach().clone()
        body_pose = art.math.rotation_matrix_to_axis_angle(init_pose).reshape(self.batch_size, -1).detach().clone()
        global_tran = init_tran.detach().clone()
        gp, joint, vertices = body_model.forward_kinematics(pose=init_pose, tran=init_tran, calc_mesh=True)
        body_3d_joint_mp = sync_mp3d_from_smpl(vertices, joint).detach().clone()

        # # Step 1: Optimize translation and body orientation
        # Optimize only translation and body orientation
        # body_pose.requires_grad = False
        # global_orient.requires_grad = True
        # global_tran.requires_grad = True
        # ori_tran_params = [global_orient, global_tran]
        # if self.use_lbfgs:
        #     ori_tran_optimizer = torch.optim.LBFGS(ori_tran_params, max_iter=self.max_iter, lr=self.step_size, line_search_fn='strong_wolfe')
        #     for i in range(self.num_iters):
        #         def closure():
        #             ori_tran_optimizer.zero_grad()
        #             smpl_output = self.smpl.forward(global_orient=global_orient, body_pose=body_pose, transl=global_tran)
        #             model_joints = smpl_output.joints[:, :24]
        #             loss = temporal_ori_tran_fitting_loss(model_joints, joints_2d, joints_conf, body_3d_joint=body_3d_joint)
        #             loss.backward()
        #             return loss
        #         ori_tran_optimizer.step(closure)
        # else:
        #     ori_tran_optimizer = torch.optim.Adam(ori_tran_params, lr=self.step_size, betas=(0.9, 0.999))
        #     for i in range(self.num_iters):
        #         smpl_output = self.smpl.forward(global_orient=global_orient, body_pose=body_pose, transl=global_tran)
        #         model_joints = smpl_output.joints[:, :24]
        #         loss = temporal_ori_tran_fitting_loss(model_joints, joints_2d, joints_conf, body_3d_joint=body_3d_joint)
        #         ori_tran_optimizer.zero_grad()
        #         loss.backward()
        #         ori_tran_optimizer.step()

        # Fix translation
        global_tran.requires_grad = True

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        # global_orient.requires_grad = True
        body_opt_params = [body_pose, global_tran]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_mp_joints] = 0.

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.max_iter, lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    # smpl_output = self.smpl.forward(global_orient=global_orient, body_pose=body_pose, transl=global_tran)
                    # pose = art.math.axis_angle_to_rotation_matrix(body_pose).reshape(self.batch_size, 24, 3, 3)
                    pose = batch_rodrigues(body_pose.view(-1, 3)).view([self.batch_size, -1, 3, 3])
                    if self.shape is not None:
                        gp, joint, vertices = body_model.forward_kinematics(pose=pose, tran=global_tran, calc_mesh=True, shape=self.shape)
                    else:
                        gp, joint, vertices = body_model.forward_kinematics(pose=pose, tran=global_tran, calc_mesh=True)
                    model_joints = sync_mp3d_from_smpl(vertices, joint)
                    loss = temporal_body_fitting_loss(body_pose, model_joints, joints_2d, joints_conf, self.pose_prior, self.cam_k, body_3d_joint_mp, self.imu_ori, gp[:, [joint_mask]])
                    loss.backward()
                    return loss
                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
            for i in range(self.num_iters):
                # smpl_output = self.smpl.forward(global_orient=global_orient, body_pose=body_pose, transl=global_tran)
                if self.shape is not None:
                    gp, joint, vertices = body_model.forward_kinematics(pose=body_pose, tran=global_tran, calc_mesh=True,
                                                                        shape=self.shape)
                else:
                    gp, joint, vertices = body_model.forward_kinematics(pose=body_pose, tran=global_tran, calc_mesh=True)
                model_joints = sync_mp3d_from_smpl(vertices, joint)
                loss = temporal_body_fitting_loss(body_pose, model_joints, joints_2d, joints_conf, self.pose_prior, self.cam_k, body_3d_joint_mp, self.imu_ori)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            # smpl_output = self.smpl.forward(global_orient=global_orient, body_pose=body_pose, transl=global_tran)
            pose = art.math.axis_angle_to_rotation_matrix(body_pose).reshape(self.batch_size, 24, 3, 3)
            if self.shape is not None:
                gp, joint, vertices = body_model.forward_kinematics(pose=pose, tran=global_tran, calc_mesh=True,
                                                                    shape=self.shape)
            else:
                gp, joint, vertices = body_model.forward_kinematics(pose=pose, tran=global_tran, calc_mesh=True)
            model_joints = sync_mp3d_from_smpl(vertices, joint)
            reprojection_loss = temporal_body_fitting_loss(body_pose, model_joints, joints_2d, joints_conf, self.pose_prior, self.cam_k, body_3d_joint_mp, self.imu_ori, gp[:, [joint_mask]], output='reprojection')

        pose = art.math.axis_angle_to_rotation_matrix(body_pose).detach()
        tran = global_tran.detach()
        return pose, tran, reprojection_loss

    def get_fitting_loss(self, pose, tran, keypoints_2d):

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_mp_joints] = 0.
        # Split SMPL pose to body pose and global orientation
        body_pose = art.math.rotation_matrix_to_axis_angle(pose).reshape(self.batch_size, -1)
        # global_orient = pose[:, :3]
        # smpl_output = self.smpl.forward(global_orient=global_orient, body_pose=body_pose, transl=global_tran)
        gp, joint, vertices = body_model.forward_kinematics(pose=pose, tran=tran, calc_mesh=True)
        body_3d_joint_mp = sync_mp3d_from_smpl(vertices, joint)
        with torch.no_grad():
            # smpl_output = self.smpl.forward(global_orient=global_orient, body_pose=body_pose, transl=global_tran)
            if self.shape is not None:
                gp, joint, vertices = body_model.forward_kinematics(pose=pose, tran=tran, calc_mesh=True,
                                                                    shape=self.shape)
            else:
                gp, joint, vertices = body_model.forward_kinematics(pose=pose, tran=tran, calc_mesh=True)
            model_joints = sync_mp3d_from_smpl(vertices, joint)
            reprojection_loss = temporal_body_fitting_loss(body_pose, model_joints, joints_2d, joints_conf, self.pose_prior, self.cam_k, body_3d_joint_mp, self.imu_ori, gp[:, [joint_mask]], output='reprojection')
        return reprojection_loss
