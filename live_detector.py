import time
import torch
import win32api
import numpy as np
import detector.cm as cm
import cv2
from pygame.time import Clock
import utils
from live_demo_sync_noitom import SyncIMUCam
import socket
import articulate as art
from net.sig_mp import Net
import config
import mediapipe as mp

# define configs
body_model = art.ParametricModel(config.paths.smpl_file)
# K = torch.tensor([[962.09465926, 0., 472.70325838], [0., 960.45287008, 357.28377582], [0., 0., 1.]])
K = torch.tensor([[623.79949084, 0., 313.69863974], [0., 623.09646347, 236.76807598], [0., 0., 1.]])
height, width = 480, 640
cam_id = 1
save_frame = 2 * 3600
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
server_ip = '127.0.0.1'


def run_detector(height, width):
    sync_imu_cam = SyncIMUCam(cam_id=cam_id, height=height, width=width)
    uv_pre, last_frame = None, None
    sync_imu_cam.clear()
    clock = Clock()
    accs, oris, RCMs, uvs  = [], [], [], []
    count = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with mp_pose.Pose(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0001,
            model_complexity=1) as mp_detector:
        while True:
            clock.tick()
            t, ori, acc, frame, RCM = sync_imu_cam.get()
            if save_frame != 0 and count < save_frame:
                accs.append(acc)
                oris.append(ori)
                RCMs.append(RCM)
                if frame is not None:
                    out.write(frame)
            if frame is not None:
                uv = torch.rand(33, 3)
                uv[:, -1] = 0.
                image = frame.copy()
                image.flags.writeable = False
                results = mp_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks is not None:
                    uv = []
                    for i in results.pose_landmarks.landmark:
                        uv.append([i.x * frame.shape[1], i.y * frame.shape[0], i.visibility])
                    uv = torch.tensor(uv)
                    cv2.rectangle(image, (uv[:, 0].min().int().item(), uv[:, 1].min().int().item()),
                                  (uv[:, 0].max().int().item(), uv[:, 1].max().int().item()), (0, 255, 0), 2)
                last_frame = image.copy()
                cv2.imshow('frame', image)
                c = cv2.waitKey(1)
                if c == ord('r'):
                    sync_imu_cam.clear()
                uv[..., :2] = K.inverse().matmul(art.math.append_one(uv[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
                uv_pre = uv.clone()
            else:
                uv = uv_pre.clone()
            uvs.append(uv)
            last_frame = cv2.imencode('.jpg', last_frame)[1]
            data = np.array(last_frame).tostring()
            uv, ori, acc, RCM = uv.numpy(), ori.numpy(), acc.numpy(), RCM.numpy()
            data = (','.join([str(i) for i in uv.reshape(-1)]) + '#' + ','.join(
                [str(i) for i in ori.reshape(-1)]) + '#' + ','.join([str(i) for i in acc.reshape(-1)]) + '#' + ','.join(
                [str(i) for i in RCM.reshape(-1)])).encode()
            s.sendto(data, (server_ip, 9999))
            count += 1
            if count == save_frame:
                out.release()
                torch.save((oris, accs, RCMs, uvs), name_file + '.pt')


if __name__ == '__main__':
    name_file = str(time.time())
    if save_frame != 0:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(name_file + '.mp4', fourcc, 30.0, (width, height))
    mp_pose = mp.solutions.pose
    run_detector(height, width)
