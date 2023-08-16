# This script is borrowed from https://github.com/nkolot/SPIN/.

import torch
import articulate as art

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

def temporal_body_fitting_loss(body_pose, model_joints, joints_2d, joints_conf, pose_prior, cam_k, body_3d_joint, imu_ori, ori, sigma=100, pose_prior_weight=0.1, angle_prior_weight=15.2, smooth_2d_weight=0.01, smooth_3d_weight=1.0, body_3d_weight=1, imu_ori_weight=0.5, output='sum'):
    """
    Loss function for body fitting
    """
    # pose_prior_weight = 1.
    # shape_prior_weight = 1.
    # angle_prior_weight = 1.
    # sigma = 10.
    # 3d loss
    body_3d_joint = body_3d_joint[:, 1:] - body_3d_joint[:, :1]
    body_3d_joint_pred = model_joints[:, 1:] - model_joints[:, :1]
    body_3d_loss = (body_3d_weight ** 2) * ((body_3d_joint_pred - body_3d_joint) ** 2).sum(dim=-1)

    projected_joints = model_joints / model_joints[..., 2:]
    projected_joints = cam_k.matmul(projected_joints.unsqueeze(-1)).squeeze(-1)[..., :2]

    imu_ori_loss = (imu_ori_weight ** 2) * ((art.math.rotation_matrix_to_axis_angle(imu_ori).reshape(body_pose.shape[0], -1) -
                                             art.math.rotation_matrix_to_axis_angle(ori).reshape(body_pose.shape[0], -1)) ** 2).sum(dim=-1)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    # joints_invalid = torch.where(joints_conf.mean(dim=1)<0.3)
    # joints_conf[joints_conf < 0.3] = 0.0
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)
    # reprojection_loss[joints_invalid] = 0.0
    pose_axis = body_pose.reshape(body_pose.shape[0], -1)[:, 3:]
    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(pose_axis, None)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(pose_axis).sum(dim=-1)

    # Regularizer to prevent betas from taking large values, we fix shape
    # shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + body_3d_loss.sum(dim=-1) + imu_ori_loss.sum(dim=-1)

    # Smooth 2d joint loss
    joint_conf_diff = joints_conf[1:]
    joints_2d_diff = projected_joints[1:] - projected_joints[:-1]
    smooth_j2d_loss = (joint_conf_diff ** 2) * joints_2d_diff.abs().sum(dim=-1)
    smooth_j2d_loss = torch.cat(
        [torch.zeros(1, smooth_j2d_loss.shape[1], device=body_pose.device), smooth_j2d_loss]
    ).sum(dim=-1)
    smooth_j2d_loss = (smooth_2d_weight ** 2) * smooth_j2d_loss

    # Smooth 3d joint loss
    joints_3d_diff = model_joints[1:] - model_joints[:-1]
    # joints_3d_diff = joints_3d_diff * 100.
    smooth_j3d_loss = (joint_conf_diff ** 2) * joints_3d_diff.abs().sum(dim=-1)
    smooth_j3d_loss = torch.cat(
        [torch.zeros(1, smooth_j3d_loss.shape[1], device=body_pose.device), smooth_j3d_loss]
    ).sum(dim=-1)
    smooth_j3d_loss = (smooth_3d_weight ** 2) * smooth_j3d_loss

    total_loss += smooth_j2d_loss + smooth_j3d_loss

    # print(f'joints: {reprojection_loss[0].sum().item():.2f}, '
    #       f'pose_prior: {pose_prior_loss[0].item():.2f}, '
    #       f'angle_prior: {angle_prior_loss[0].item():.2f}, '
    #       f'smooth_j2d: {smooth_j2d_loss.sum().item()}, '
    #       f'smooth_j3d: {smooth_j3d_loss.sum().item()}',
    #       f'body_3d: {body_3d_loss[0].sum().item():.2f}'
    #       f'imu_ori: {imu_ori_loss[0].sum().item():.2f}')

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def temporal_ori_tran_fitting_loss(model_joints, joints_2d, joints_conf, body_3d_joint, body_3d_loss_weight=1000):
    """
    Loss function for orientation and translation optimization.
    """
    # 3d loss
    # body_3d_joint = body_3d_joint[:, 1:] - body_3d_joint[:, :1]
    # body_3d_joint_pred = model_joints[:, 1:] - model_joints[:, :1]
    # Project model joints
    projected_joints = model_joints / model_joints[..., 2:]
    projected_joints = projected_joints[..., :2]
    # l_shoulder, r_shoulder, l_hip, r_hip
    op_smpl_joints_ind, op_mp_joints_ind = [16, 17, 1, 2], [11, 12, 23, 24]
    reprojection_error_op = (joints_2d[:, op_mp_joints_ind] - projected_joints[:, op_smpl_joints_ind]) ** 2
    is_valid = (joints_conf[:, op_mp_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op).sum(dim=(1, 2))
    body_3d_loss = (body_3d_joint[:, op_smpl_joints_ind] - model_joints[:, op_smpl_joints_ind]) ** 2
    total_loss = reprojection_loss + body_3d_loss_weight * body_3d_loss.sum(dim=(1, 2))
    # print('reprojection_loss: ', reprojection_loss.sum().cpu().item())
    # print('body_3d_loss: ', body_3d_loss.sum(dim=(1, 2)).sum().cpu().item())
    return total_loss.sum()
