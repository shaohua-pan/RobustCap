import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
import mediapipe as mp
import config
import pickle
from scripts.smooth_bbox import get_smooth_bbox_params

device = "cuda" if torch.cuda.is_available() else "cpu"
pt_path = os.path.join(config.paths.pw3d_raw_dir, 'kp2d_mp')
mp_detector = mp.solutions.pose.Pose(static_image_mode=True, enable_segmentation=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
data_path = config.paths.pw3d_raw_dir

def detect_mp(debug=True, border=60, vis_mp=True):
    sequences = [x.split('.')[0] for x in os.listdir(os.path.join(data_path, 'sequenceFiles', 'test'))]
    for name in tqdm(sequences):
        data = pickle.load(open(os.path.join(data_path, 'sequenceFiles', 'test', name + '.pkl'), 'rb'), encoding='latin1')
        img_dir = os.path.join(data_path, 'imageFiles', name)
        num_people = len(data['poses'])
        for p_id in range(num_people):
            uvs = []
            if os.path.exists(os.path.join(pt_path, name + '_' + str(p_id) + '.pt')):
                continue
            j2d = data['poses2d'][p_id].transpose(0, 2, 1)
            im_names = sorted(os.listdir(img_dir))
            # last_visible = None
            bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2d, vis_thresh=0.3, sigma=8)
            c_x = bbox_params[:, 0].astype(np.int32)
            c_y = bbox_params[:, 1].astype(np.int32)
            scale = bbox_params[:, 2]
            im = cv2.imread(os.path.join(img_dir, im_names[0]))
            if num_people != 1 or im.shape[0] > im.shape[1]:
                w = h = 100. / scale
                h = h * 1.8
            else:
                w = h = 150. / scale
                w = h = h * 1.1
            w = w.astype(np.int32)
            h = h.astype(np.int32)
            for index, im_name in enumerate(im_names):
                im = cv2.imread(os.path.join(img_dir, im_name))
                if j2d[index, :, 2].mean() < 0.3:
                    uvs.append(None)
                    continue
                sx, sy, ex, ey = int(max(0, c_x[index] - w[index] // 2)), int(max(0, c_y[index] - h[index] // 2)), int(min(c_x[index] + w[index] // 2, im.shape[1])), int(min(c_y[index] + h[index] // 2, im.shape[0]))
                im = im[sy:ey, sx:ex]
                with mp_pose.Pose(
                        model_complexity=1,
                        min_detection_confidence=0.0) as pose:
                    results = pose.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks is None:
                        uvs.append(None)
                    else:
                        uv = []
                        for i in results.pose_landmarks.landmark:
                            uv.append([i.x*im.shape[1]+sx, i.y*im.shape[0]+sy, i.visibility])
                        uvs.append(torch.tensor(uv))
                    if debug and vis_mp:
                        if results.pose_landmarks is not None:
                            mp_drawing.draw_landmarks(
                                im,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                            cv2.imshow('img', im)
                            cv2.waitKey(1)
                    elif debug:
                        if uv is not None:
                            for kp in uv:
                                frame = cv2.circle(frame, (int(kp[0]), int(kp[1])), radius=5, color=[0, 165, 255], thickness=-1)
                            cv2.imshow('img', frame)
                            cv2.waitKey(1)
            print(f'saving{name}_person{p_id}')
            os.makedirs(pt_path, exist_ok=True)
            torch.save(uvs, os.path.join(pt_path, name + '_' + str(p_id) + '.pt'))


if __name__ == '__main__':
    detect_mp()