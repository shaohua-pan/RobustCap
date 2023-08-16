import torch
import articulate as art
from .temporal_smplify import TemporalSMPLify
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def smplify_runner(
        pred_pose,
        pred_tran,
        j2dc,
        imu_ori,
        batch_size,
        cam_k,
        lr=1.0,
        opt_steps=1,
        use_lbfgs=True,
        loss_threshold=20000,
        shape=None,
        use_head=False,
):
    cam_k = cam_k.to(device)
    smplify = TemporalSMPLify(step_size=lr, batch_size=batch_size, num_iters=opt_steps, use_lbfgs=use_lbfgs, cam_k=cam_k, imu_ori=imu_ori, shape=shape, use_head=use_head)
    pred_pose = pred_pose.reshape(batch_size, -1).to(device)
    pred_tran = pred_tran.reshape(-1, 3).to(device)
    j2dc = j2dc.reshape(-1, 33, 3).to(device)
    gt_keypoints_2d_orig = j2dc
    # Before running compute reprojection error of the network
    opt_joint_loss = smplify.get_fitting_loss(pred_pose.detach(), pred_tran, gt_keypoints_2d_orig).mean(dim=-1)
    if opt_joint_loss[0].sum().cpu().item() > loss_threshold:
        return pred_pose.cpu().reshape(-1, 24, 3, 3), pred_tran.cpu().reshape(-1, 3), None
    # Run SMPLify optimization initialized from the network prediction
    pose, tran, new_opt_joint_loss = smplify(pred_pose.detach(), pred_tran.detach(), gt_keypoints_2d_orig)
    new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
    # Will update the dictionary for the examples where the new loss is less than the current one
    update = (new_opt_joint_loss < opt_joint_loss)
    return pose.cpu().reshape(-1, 24, 3, 3), tran.cpu().reshape(-1, 3), update