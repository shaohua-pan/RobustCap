import time
import torch
import numpy as np
import cv2
from pygame.time import Clock
import utils
from live_demo_sync import SyncIMUCam
import socket
import articulate as art
import config
import mediapipe as mp

# define configs
body_model = art.ParametricModel(config.paths.smpl_file)
K = torch.tensor(config.Live.camera_intrinsic)
height, width = config.Live.camera_height,config.Live.camera_width
cam_id = config.Live.camera_id
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
server_ip = '127.0.0.1'

def run_detector(height, width):
    sync_imu_cam = SyncIMUCam(config.Live.imu_addrs, cam_id=cam_id, height=height, width=width)
    uv_pre, last_frame = None, None
    sync_imu_cam.clear()
    clock = Clock()
    count = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with mp_pose.Pose(
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0001,
            model_complexity=1) as mp_detector:
        while True:
            clock.tick()
            t, ori, acc, frame, RCM = sync_imu_cam.get()
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
                cv2.imshow('frame', image)
                c = cv2.waitKey(1)
                if c == ord('r'):
                    sync_imu_cam.clear()
                uv[..., :2] = K.inverse().matmul(art.math.append_one(uv[..., :2]).unsqueeze(-1)).squeeze(-1)[..., :2]
                uv_pre = uv.clone()
            else:
                uv = uv_pre.clone()
            uv, ori, acc, RCM = uv.numpy(), ori.numpy(), acc.numpy(), RCM.numpy()
            data = (','.join([str(i) for i in uv.reshape(-1)]) + '#' + ','.join(
                [str(i) for i in ori.reshape(-1)]) + '#' + ','.join([str(i) for i in acc.reshape(-1)]) + '#' + ','.join(
                [str(i) for i in RCM.reshape(-1)])).encode()
            s.sendto(data, (server_ip, 9999))
            count += 1


if __name__ == '__main__':
    mp_pose = mp.solutions.pose
    run_detector(height, width)
