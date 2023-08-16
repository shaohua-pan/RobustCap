import cv2
import numpy as np
import tqdm
import torch
import config

mp_mask = torch.tensor(config.mp_mask)

def view_2d_keypoint(keypoints, parent=None, images=None, thickness=None, fps=60):
    r"""
    View 2d keypoint sequence in image coordinate frame. Modified from vctoolkit.render_bones_from_uv.

    Notes
    -----
    If num_frame == 1, only show one picture.
    If parent is None, do not render bones.
    If images is None, use 1080p white canvas.
    If thickness is None, use a default value.
    If keypoints in shape [..., 2], render keypoints without confidence.
    If keypoints in shape [..., 3], render confidence using alpha of colors (more transparent, less confident).

    :param keypoints: Tensor [num_frames, num_joints, *] where *=2 for (u, v) and *=3 for (u, v, confidence).
    :param parent: List in length [num_joints]. e.g., [None, 0, 0, 0, 1, 2, 3 ...]
    :param images: Numpy uint8 array that can expand to [num_frame, height, width, 3].
    :param thickness: Thickness for points and lines.
    :param fps: Sequence FPS.
    """
    if len(keypoints[0].shape) == 2:
        keypoints = keypoints[:, None, :, :]
    if images is None:
        images = [np.ones((keypoints.shape[1], 480, 360, 3), dtype=np.uint8) * 255 for _ in range(keypoints.shape[0])]
    if images[0].dtype != np.uint8:
        raise RuntimeError('images must be uint8 type')
    if thickness is None:
        thickness = int(max(round(images[0].shape[1] / 160), 1))
    for i in range(keypoints.shape[0]):
        images[i] = np.broadcast_to(images[i], (keypoints.shape[1], images[i].shape[-3], images[i].shape[-2], 3))
    has_conf = keypoints.shape[-1] == 3
    is_single_frame = len(images[0]) == 1

    if not is_single_frame:
        writer = cv2.VideoWriter('a.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps,
                                 (keypoints.shape[0] * images[0].shape[2], images[0].shape[1]))
    for i in tqdm.trange(len(images[0])):
        bgs = []
        for j in range(keypoints.shape[0]):
            bg = images[j][i]
            for uv in keypoints[j][i]:
                conf = float(uv[2]) if has_conf else 1
                fg = cv2.circle(bg.copy(), (int(uv[0]), int(uv[1])), int(thickness * 2), (0, 0, 255), -1)
                bg = cv2.addWeighted(bg, 1 - conf, fg, conf, 0)
            if parent is not None:
                for c, p in enumerate(parent):
                    if p is not None:
                        start = (int(keypoints[j][i][p][0]), int(keypoints[j][i][p][1]))
                        end = (int(keypoints[j][i][c][0]), int(keypoints[j][i][c][1]))
                        conf = min(float(keypoints[j][i][c][2]), float(keypoints[j][i][p][2])) if has_conf else 1
                        fg = cv2.line(bg.copy(), start, end, (255, 0, 0), thickness)
                        bg = cv2.addWeighted(bg, 1 - conf, fg, conf, 0)
            bgs.append(bg)
        bg = np.concatenate(bgs, axis=1)
        cv2.imshow('2d keypoint', bg)
        if is_single_frame:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
            writer.write(bg)
    if not is_single_frame:
        writer.release()
    cv2.destroyWindow('2d keypoint')


def view_2d_keypoint_on_z_1(keypoints, parent=None, thickness=None, scale=1, fps=60):
    r"""
    View 2d keypoint sequence on z=1 plane.

    Notes
    -----
    If num_frame == 1, only show one picture.
    If parent is None, do not render bones.
    If thickness is None, use a default value.
    If keypoints in shape [..., 2], render keypoints without confidence.
    If keypoints in shape [..., 3], render confidence using alpha of colors (more transparent, less confident).

    :param keypoints: Tensor [num_seq, num_frames, num_joints, *] where *=2 for (x, y) and *=3 for (x, y, confidence).
    :param parent: List in length [num_joints]. e.g., [None, 0, 0, 0, 1, 2, 3 ...]
    :param thickness: Thickness for points and lines.
    :param scale: Scale of the keypoints.
    :param fps: Sequence FPS.
    """
    f = 500 * scale
    assert isinstance(keypoints, list)
    keypoints = torch.stack(keypoints).clone()
    keypoints[..., 0] = keypoints[..., 0] * f + 360 / 2
    keypoints[..., 1] = keypoints[..., 1] * f + 480 / 2
    view_2d_keypoint(keypoints, parent=parent, thickness=thickness, fps=fps)


def get_bbox(uv, height, width, border=130, w_h=0.75):
    u_max, v_max, u_min, v_min = int(max(uv[:, 0])), int(max(uv[:, 1])), int(min(uv[:, 0])), int(
        min(uv[:, 1]))
    u_center, v_center = (u_max + u_min) // 2, (v_max + v_min) // 2
    # crop h:w = 4:3
    if (u_max - u_min) * w_h > (v_max - v_min):
        height_fix = (u_max - u_min) + border
        if height_fix > height:
            height_fix = height
        width_fix = int(height_fix * w_h)
    else:
        width_fix = (v_max - v_min) + border
        if width_fix > width:
            width_fix = width
        height_fix = width_fix // w_h
    if v_center - width_fix // 2 < 0:
        v_start, v_end = 0, width_fix
    elif v_center + width_fix // 2 >= width:
        v_start, v_end = width - width_fix, width
    else:
        v_start, v_end = v_center - width_fix // 2, v_center + width_fix // 2
    if u_center - height_fix // 2 < 0:
        u_start, u_end = 0, height_fix
    elif u_center + height_fix // 2 >= height:
        u_start, u_end = height - height_fix, height
    else:
        u_start, u_end = u_center - height_fix // 2, u_center + height_fix // 2
    return int(u_start), int(v_start), int(u_end), int(v_end)


def sync_mp3d_from_smpl(vert, joint):
    syn_3d = vert[:, mp_mask]
    syn_3d[:, 11:17] = joint[:, 16:22].clone()
    syn_3d[:, 23:25] = joint[:, 1:3].clone()
    syn_3d[:, 25:27] = joint[:, 4:6].clone()
    syn_3d[:, 27:29] = joint[:, 7:9].clone()
    return syn_3d


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re
