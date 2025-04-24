import cv2
import os
import torch
import mediapipe as mp
import config

# define your own path
device = "cuda" if torch.cuda.is_available() else "cpu"
video_path = os.path.join(config.paths.totalcapture_raw_dir, 'video2')
pt_path = os.path.join(config.paths.totalcapture_raw_dir, 'kp2d_mp')
mp_detector = mp.solutions.pose.Pose(static_image_mode=True, enable_segmentation=True)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def detection_mp(debug=True):
    for subject in sorted(os.listdir(video_path)):
        for motion in sorted(os.listdir(os.path.join(video_path, subject))):
            if subject == 's5' and motion == 'acting3':
                continue
            for file in sorted(os.listdir(os.path.join(video_path, subject, motion))):
                try:
                    video = subject + '_' + motion + '_' + file.split('.')[0].split('_')[-1]
                    if os.path.exists(os.path.join(pt_path, f'{video}.pt')):
                        print('exists pt' + video)
                        continue
                    cap = cv2.VideoCapture(os.path.join(video_path, subject, motion, file))
                    uvs = []
                    with mp_pose.Pose(
                            min_detection_confidence=0.0,
                            min_tracking_confidence=0.5,
                            model_complexity=1) as pose:
                        while True:
                            rval, image = cap.read()
                            if image is None:
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
                    print(f'saving{video}')
                    os.makedirs(pt_path, exist_ok=True)
                    torch.save(uvs, os.path.join(pt_path, video + '.pt'))
                except Exception as e:
                    print(e)
                    print('wrong processing video, continue next')


if __name__ == '__main__':
    detection_mp(debug=False)
