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

# define configs
device = "cuda" if torch.cuda.is_available() else "cpu"
body_model = art.ParametricModel(config.paths.smpl_file)
unity_exe = r'C:\Users\thucg\Desktop\live\ci.exe'
server_ip = '127.0.0.1'


def convert_from_str(x):
    x = x.split(',')
    data = []
    for i in x:
        data.append(float(i))
    return np.asarray(data)

def run_live_demo(net):
    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    # win32api.ShellExecute(0, 'open', unity_exe, '', '', 1)
    conn, addr = server_for_unity.accept()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((server_ip, 9999))
    data, addr = s.recvfrom(4000000)
    uv, ori, acc, RCM = data.decode().split('#')
    RCM = torch.from_numpy(convert_from_str(RCM)).reshape(3, 3).float()
    net.gravityc = torch.matmul(RCM, torch.tensor([0., -1, 0.]).unsqueeze(-1)).squeeze(-1)
    clock = Clock()
    f = 0
    count = 0
    stran = None
    while True:
        clock.tick()
        data, addr = s.recvfrom(4000000)
        uv, ori, acc, _ = data.decode().split('#')
        uv, ori, acc = torch.from_numpy(convert_from_str(uv)).reshape(33, 3).float(), torch.from_numpy(convert_from_str(ori)).reshape(6, 3, 3).float(), torch.from_numpy(convert_from_str(acc)).reshape(6, 3).float()
        # im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        # cv2.imshow('frame', im)
        # cv2.waitKey(1)
        if stran is None:
            pose, tran = net.forward_online(uv.to(device), acc.to(device), ori.to(device), first_frame=True)
        else:
            pose, tran = net.forward_online(uv.to(device), acc.to(device), ori.to(device))
        root = RCM.T.matmul(pose[0])
        pose[0] = root
        tran = RCM.T.matmul(tran.unsqueeze(-1)).squeeze(-1)
        if stran is None:
            stran = tran.clone()
        tran = tran - stran
        pose = art.math.rotation_matrix_to_axis_angle(pose).view(-1)
        f += 1
        unity_data = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        conn.send(unity_data.encode('utf8'))
        count += 1


if __name__ == '__main__':
    net = Net().to(device)
    net.live = True
    conf_range = (0.85, 0.9)
    tran_filter_num = 0.01
    net.load_state_dict(torch.load('./data/weights/sig_mp/best_weights.pt'))
    net.eval()
    run_live_demo(net)
