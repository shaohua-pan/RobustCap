import cv2
import numpy as np
import os
import time
import glob
import torch

import config
from tqdm import tqdm
from aist_plusplus.loader import AISTDataset
import utils
import os
import mediapipe as mp
import scripts.occlusion as occlusion
import random

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
device = "cuda" if torch.cuda.is_available() else "cpu"
# define your own path of dataset
video_path = r'F:\ShaohuaPan\dataset\AIST\all_video'
pt_mp_path = 'F:\ShaohuaPan\dataset\AIST\kp_mp'
pt_mp_occ_path = 'F:\ShaohuaPan\dataset\AIST\kp_mp_occ'
name_path = os.path.join(config.paths.aist_raw_dir, 'splits/pose_val.txt')
mp_detector = mp.solutions.pose.Pose(static_image_mode=True, enable_segmentation=True)
f = open(name_path, "r")
seq_names_all = f.readlines()
f.close()
seq_names = [seq_name.strip() for seq_name in seq_names_all]
seq_names = [seq_name.split('.')[0] for seq_name in seq_names]


def detection_mediapipe(debug=True):
    for seq_name in tqdm(seq_names):
        for view in AISTDataset.VIEWS:
            video_name = AISTDataset.get_video_name(seq_name, view)
            video_file = os.path.join(video_path, video_name + ".mp4")
            if not os.path.exists(video_file):
                print('not find' + video_file)
                continue
            if os.path.exists(os.path.join(pt_mp_path, f'{video_name}.pt')):
                print('exists pt' + video_file)
                continue
            cap = cv2.VideoCapture(video_file)
            uvs = []
            with mp_pose.Pose(
                    min_detection_confidence=0.0,
                    min_tracking_confidence=0.5,
                    model_complexity=1) as pose:
                while True:
                    rval, image = cap.read()
                    if not rval:
                        break
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    # Draw the pose annotation on the image.
                    if debug:
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        cv2.imshow('MediaPipe Pose', image)
                        cv2.waitKey(1)
                    if results.pose_landmarks is None:
                        uvs.append(None)
                    else:
                        uv = []
                        for i in results.pose_landmarks.landmark:
                            uv.append([i.x, i.y, i.z, i.visibility])
                        uvs.append(torch.tensor(uv))
                print(f'saving{video_name}')
                os.makedirs(pt_mp_path, exist_ok=True)
                torch.save(uvs, os.path.join(pt_mp_path, video_name + '.pt'))


def detection_mediapipe_occ(debug=False):
    occluders = occlusion.load_occluders(config.paths.occ_dir)
    for seq_name in tqdm(seq_names):
        filter = np.random.randint(1, 9, 4).tolist()
        for index, view in enumerate(AISTDataset.VIEWS):
            video_name = AISTDataset.get_video_name(seq_name, view)
            video_file = os.path.join(video_path, video_name + ".mp4")
            if not os.path.exists(video_file):
                print('not find' + video_file)
                continue
            if os.path.exists(os.path.join(pt_mp_occ_path, f'{video_name}.pt')):
                print('exists pt' + video_file)
                continue
            cap = cv2.VideoCapture(video_file)
            uvs = []
            width_height = np.asarray([1920, 1080])
            im_scale_factor = min(width_height) / 256
            count = np.random.randint(1, 8)
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
                    min_detection_confidence=0.0,
                    min_tracking_confidence=0.5,
                    model_complexity=1) as pose:
                while True:
                    rval, image = cap.read()
                    if not rval:
                        break
                    for _ in range(count):
                        occlusion.paste_over(im_src=occs[_], im_dst=image, center=centers[_])
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    # Draw the pose annotation on the image.
                    if debug:
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        cv2.imshow('MediaPipe Pose', image)
                        cv2.waitKey(1)
                    if results.pose_landmarks is None:
                        uvs.append(None)
                    else:
                        uv = []
                        for i in results.pose_landmarks.landmark:
                            uv.append([i.x, i.y, i.z, i.visibility])
                        uvs.append(torch.tensor(uv))
                print(f'saving{video_name}')
                os.makedirs(pt_mp_occ_path, exist_ok=True)
                torch.save(uvs, os.path.join(pt_mp_occ_path, video_name + '.pt'))


if __name__ == '__main__':
    detection_mediapipe()
    detection_mediapipe_occ()