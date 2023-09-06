# RobustCap

Code for our SIGGRAPH ASIA 2023 [paper](https://arxiv.org/abs/2309.00310) "Fusing Monocular Images and Sparse IMU Signals for Real-time Human

Motion Capture". This repository contains the system implementation and evaluation. See [Project Page](https://shaohua-pan.github.io/robustcap-page/).
<div align="left">
<img src="assets/occlusion.gif" width="320" height="180"> <img src="assets/sports.gif" width="320" height="180"> <img src="assets/dark.gif" width="320" height="180">
<img src="assets/comparison.gif" width="320" height="180">
<br>
</div>

## Installation
```
conda create -n RobustCap python=3.8
conda activate RobustCap
pip install -r requirements.txt
```
Install pytorch cuda version from the official [website](https://pytorch.org/).
## Data
### SMPL Files, Pretrained Model and Test Data
- Download smpl files from [here](https://drive.google.com/file/d/1lsHC3mupzGqrzHEkXlXwKWXtw5d8Fxr3/view?usp=drive_link) or the official [website](https://smpl.is.tue.mpg.de/). Unzip it and place it at `models/`. 
- Download the [pretrained model and data](https://drive.google.com/file/d/1oDnFd8h4mTCSYKD4zEA0AL3b6qUeUtvl/view?usp=drive_link) and place them at `data/`.
- For AIST++ evaluation, download the [no aligned files](https://drive.google.com/file/d/12RSdlg1Px0EUgZKybqY-exUJWK9HskAD/view?usp=drive_link) and place it at `data/dataset_work/AIST`.
## Evaluation
We provide the evaluation code for AIST++, TotalCapture, 3DPW and 3DPW-OCC. The results maybe slightly different from the numbers reported in the paper due to the randomness of the optimization.
```
python evaluate.py
```
## Visualization
We provide the visualization code for AIST++. You can use view_aist function in evaluate.py to visualize the results. By indicating seq_idx and cam_idx, you can visualize the results of a specific sequence and camera. Set vis=True to visualize the overlay results. Use body_model.view_motion to visualize the open3d results.

## Todo
- Visualization.
- Live demo code.
## Citation  
```
TBA
```
