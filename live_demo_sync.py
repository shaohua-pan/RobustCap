from queue import Queue
import cv2
import time
import socket
import threading
import numpy as np
import torch
from pygame.time import Clock
import articulate as art
from articulate.utils.print import *
import select
import os
import atexit
from articulate.utils.bullet import RotationViewer


class SyncIMUCam:
    def __init__(self, imus_addr, cam_id=1, fps=60, sync_cam=False, height=720, width=960):
        print_yellow('=========== Connecting Xsens dot sensors ==========')
        atexit.register(self._exit)
        self.N = len(imus_addr)
        self.cs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cs.bind(('127.0.0.1', 8777))
        self.cs.setblocking(True)
        self.cs.settimeout(5)

        print_green('succeed')

        print_yellow('=========== Connecting camera ==========')
        # self.cam = PhoneCamera(cam_port)
        self.cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cam.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.sync_cam = sync_cam
        print_green('succeed')

        self.sync_offset = self.jump_synchronization(debug=True)
        self.RMI, self.RSB, self.RCI, self.RCM = self.tpose_calibration()

        self.t = [0 for _ in range(2)]
        self.internal_time = 2
        self.dt = 1 / fps
        self.sync_measurements = Queue(maxsize=3000)
        self.running = True
        self.clear()
        self.thread = threading.Thread(target=self.run)
        self.thread.setDaemon(True)
        self.thread.start()

    def _exit(self):
        print('exiting ...')
        if hasattr(self, 'thread'):
            print('\twaiting thread join')
            self.running = False
            self.thread.join()
        if hasattr(self, 'cam'):
            print('\tclosing camera')
            del self.cam
        os.system('taskkill /f /im Hybrid.exe')
        print('finish')

    def tpose_calibration(self):
        print_yellow('========== T-pose calibration started ==========')
        c = input('Used cached RMI? [y]/n    (If you choose no, align imu 0 with body (x = Forward, y = Left, z = Up).')
        if c == 'n' or c == 'N':
            print('Keep for 2 seconds ...')
            self.clear()
            qs = torch.stack([self.get_from_udp()[1][0] for _ in range(60 * 2)])
            qs = art.math.quaternion_mean(qs)
            RSI = art.math.quaternion_to_rotation_matrix(qs).view(3,
                                                                  3).t()  # M = model, B = bone, S = sensor, I = inertial
            RMS = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0.]])
            RMI = RMS.mm(RSI)
            torch.save(RMI, 'data/temp/RMI.pt')
        else:
            RMI = torch.load('data/temp/RMI.pt')
        c = input('Used cached RCI? [y]/n    (If you choose no, align imu 0 with camera (x = Up, y = Right, z = Forward).')
        if c == 'n' or c == 'N':
            print('Keep for 2 seconds ...')
            self.clear()
            qs = torch.stack([self.get_from_udp()[1][0] for _ in range(60 * 2)])
            qs = art.math.quaternion_mean(qs)
            RIS = art.math.quaternion_to_rotation_matrix(qs).view(3, 3)
            RSC = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
            RCI = RIS.mm(RSC).t()
            torch.save(RCI, 'data/temp/RCI.pt')
        else:
            RCI = torch.load('data/temp/RCI.pt')
        input('Wear all imus correctly and press any key.')
        for i in range(3, 0, -1):
            print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
            time.sleep(1)
        print('\rStand straight in T-pose. Keep the pose for 3 seconds ...')
        self.clear()
        time.sleep(2)
        qs, accs = [], []
        qmean = []
        for _ in range(60 * 2):
            t, q, a = self.get_from_udp()
            qmean.append(q)
        for i in range(self.N):
            qs.append(art.math.quaternion_mean(torch.stack(qmean)[:, i, :]))
        RIS = art.math.quaternion_to_rotation_matrix(torch.stack(qs))
        RSB = RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))  # = (R_MI R_IS)^T R_MB = R_SB
        RCM = RCI.matmul(RMI.T)
        print_green('T-pose calibration finished')
        return RMI, RSB, RCI, RCM

    def jump_synchronization(self, debug=False):
        if debug:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((400, 200))
            pygame.display.set_caption('Jump Synchronization Debug')
            ax, ays = [0.], [[0.] for _ in range(self.N + 1)]

        print_yellow('========== Jump synchronization started ==========')
        jump_timestamps = [[] for _ in range(self.N + 1)]
        old_sync, oldim = None, None
        reset_cnt = 0
        set_cnt = 0

        while True:
            t = time.time()
            self.cam.read()
            if time.time() - t > 1 / 40:  # queue empty
                break
        self.clear()
        print('Waiting for a jump ...')
        for k in range(60 * 60 * 3):
            debug_plot = debug and k % 4 == 2
            if debug_plot:
                screen.fill((255, 255, 255))
                ax.append(k)
            ts, qs, a_s = self.get_from_udp()
            for i in range(self.N):
                t, q, a = ts[i], qs[i], a_s[i]
                if debug_plot:
                    ays[i].append(a.norm().item())
                    points = [(j * 4, 200 - v * 10) for j, v in enumerate(ays[i][-100:])]
                    pygame.draw.lines(screen, (i * 40, 0, 255 - i * 40), False, points, width=1)
                if a.norm() > 9:
                    jump_timestamps[i].append(t)
            if k % 2 == 0:
                _, im = self.cam.read()
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                flag = 200 / (cv2.Laplacian(im, cv2.CV_64F).var() + 1)
                if debug_plot:
                    ays[-1].append(float(flag))
                    points = [(j * 4, 200 - v * 10) for j, v in enumerate(ays[-1][-100:])]
                    pygame.draw.lines(screen, (255, 0, 0), False, points, width=1)
                if flag > 7:  # need modification in new environments
                    jump_timestamps[-1].append(self.cam.get(cv2.CAP_PROP_POS_MSEC) / 1000)

            if self.sync_cam:
                is_jump = [len(_) > 0 for _ in jump_timestamps]
            else:
                is_jump = [len(_) > 0 for _ in jump_timestamps[:-1]]

            print('\r', is_jump, end='        ')
            if any(is_jump):
                reset_cnt += 1
            if all(is_jump):
                set_cnt += 1
            if set_cnt > 60:
                print('\ndetect a jump: ', end='')
                if self.sync_cam:
                    sync = torch.tensor([(_[0] + _[-1]) / 2 for _ in jump_timestamps])
                else:
                    sync = torch.tensor([(_[0] + _[-1]) / 2 for _ in jump_timestamps[:-1]])
                print(sync, end='\t')
                if old_sync is not None:
                    err = (sync - sync[0] - old_sync + old_sync[0]).abs().max().item()
                    print('maxerr=%.4fs\t' % err, 'succeed' if err < 0.04 else 'jump again')
                    if err < 0.4:  # ###################################0.04
                        old_sync = sync
                        break
                else:
                    print('jump again')
                old_sync = sync
                jump_timestamps = [[] for _ in range(self.N + 1)]
                reset_cnt = 0
                set_cnt = 0
            if reset_cnt > 120:
                jump_timestamps = [[] for _ in range(self.N + 1)]
                reset_cnt = 0
                set_cnt = 0

            if debug_plot:
                pygame.display.update()
                cv2.imshow('camera', im)
                if cv2.waitKey(1) == ord('r'):
                    print('\nreset')
                    while True:
                        t = time.time()
                        self.cam.read()
                        if time.time() - t > 1 / 40:  # queue empty
                            break
                    self.clear()
        if debug:
            cv2.destroyAllWindows()
            pygame.quit()
        print_green('Jump synchronization finished')
        return old_sync

    def run(self):
        self.RMI = self.RMI.cuda()
        self.RSB = self.RSB.cuda()
        self.RCI = self.RCI.cuda()
        frame = 0
        while self.running:
            frame += 1
            imus_q, imus_a, im = [], [], None
            ts, qs, a_s = self.get_from_udp()
            t = ts[0] - self.sync_offset[0]
            while self.internal_time + self.dt < t:
                # print('warning: skip a tick')
                self.internal_time += self.dt
            while True:
                if self.internal_time <= t:
                    break
                else:
                    ts, qs, a_s = self.get_from_udp()
                    t = ts[0] - self.sync_offset[0]
            for i in range(self.N):
                imus_q.append(qs[i])
                imus_a.append(a_s[i])
            now = time.time()
            if self.sync_cam and self.internal_time > self.t[-1] + self.dt:
                _, im = self.cam.read()
                self.t[-1] = self.cam.get(cv2.CAP_PROP_POS_MSEC) / 1000 - self.sync_offset[-1]
            elif frame % 2 == 0:
                _, im = self.cam.read()
                self.t[-1] = self.cam.get(cv2.CAP_PROP_POS_MSEC) / 1000
            RIS = art.math.quaternion_to_rotation_matrix(torch.stack(imus_q)).cuda()
            RCB = self.RCI.matmul(RIS).matmul(self.RSB)
            aC = torch.stack(imus_a).cuda().mm(self.RCI.t())
            if self.sync_measurements.full():
                print('warning: queue is full')
                self.sync_measurements.get()
            self.sync_measurements.put((self.internal_time, RCB, aC, im, self.RCM))
            # self.internal_time += self.dt
            self.internal_time += max(self.dt, (time.time() - now))

    def get(self):
        while True:
            try:
                return self.sync_measurements.get(timeout=1)
            except Exception:
                print('err')

    def clear(self):
        self.sync_measurements = Queue(maxsize=3000)
        while True:
            ready = select.select([self.cs], [], [], 0.01)
            if ready[0]:
                self.cs.recv(int(32 * self.N))
            else:
                return

    def get_from_udp(self):
        data, _ = self.cs.recvfrom(32 * self.N)
        data = np.frombuffer(data, np.float32).copy()
        t = data[:self.N]
        q = data[self.N:5 * self.N].reshape(self.N, 4)
        a = data[5 * self.N:].reshape(self.N, 3)
        return t.tolist(), torch.from_numpy(q), torch.from_numpy(a)


if __name__ == '__main__':
    imus_addr = [
        'D4:22:CD:00:36:03',
    ]
    sync_imu_cam = SyncIMUCam(imus_addr)
    from pygame.time import Clock

    clock = Clock()
    with RotationViewer(1) as viewer:
        sync_imu_cam.clear()
        while True:
            clock.tick()
            _, R, a, im, _ = sync_imu_cam.get()
            if im is not None:
                cv2.imshow('image', im)
                c = cv2.waitKey(1)
                if c == ord('r'):
                    sync_imu_cam.clear()
            viewer.update_all(R)
            print('\r', clock.get_fps(), sync_imu_cam.sync_measurements.qsize(), end='')
