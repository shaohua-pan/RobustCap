import os
if __name__ == '__main__':
    import sys

    os.chdir('..')
    sys.path.insert(0, os.getcwd())

import torch
import articulate as art
from articulate.utils.torch import *
from config import *
from torch.nn.functional import relu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(paths.smpl_file, device=device)
body_model_cpu = art.ParametricModel(paths.smpl_file)
mp_mask = torch.tensor(mp_mask)

class Net(torch.nn.Module):
    """
    The network for the motion prediction.
    """
    hidden_size = 512
    conf_range = (0.7, 0.8)
    contact_threshold = 0.7
    smooth = 1
    use_flat_floor = True
    use_reproj_opt = False
    use_vision_updater = True
    use_imu_updater = True
    name = 'sig_mp'
    gravityc = torch.tensor([-0.0029, 0.9980, -0.0273])
    imu_num = 6
    height_threhold = 0.15
    distrance_threshold = 10
    tran_filter_num = 0.05

    live = False
    if live:
        conf_range = (0.85, 0.9)
        tran_filter_num = 0.01
    update_vision_freq = 30
    update_vision_count = 0
    j_temp = None

    def __init__(self):
        """
        Initialize the network.
        """
        super(Net, self).__init__()
        self.rnn2 = RNNWithInit(input_size=Net.imu_num * 3 + Net.imu_num * 9,
                        output_size=23 * 3,
                        hidden_size=Net.hidden_size,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn3 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 23 * 3,
                        output_size=3,
                        hidden_size=Net.hidden_size,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn4 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 33 * 3,
                        output_size=23 * 3,
                        hidden_size=1024+256,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn6 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 33 * 3 + 23 * 3,
                        output_size=3,
                        hidden_size=1024,
                        num_rnn_layer=2,
                        dropout=0.4)
        self.rnn7 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 23 * 3,
                        output_size=24 * 6,
                        hidden_size=512,
                        num_rnn_layer=2,
                        dropout=0.1)
        self.rnn8 = RNN(input_size=Net.imu_num * 3 + Net.imu_num * 9 + 23 * 3,
                        output_size=2,
                        hidden_size=Net.hidden_size,
                        num_rnn_layer=2,
                        dropout=0.4)

        j = body_model.get_zero_pose_joint_and_vertex()[0]
        self.b = body_model.joint_position_to_bone_vector(j.unsqueeze(0)).view(24, 3, 1).to(device)
        self.hidden = [None for _ in range(8)]
        self.temp_hidden = None
        self.last_pfoot = None
        self.last_tran = None
        self.floor_y = []
        self.first_reach = True

    def reset_states(self):
        """
        Reset the hidden states and necessary variables.
        """
        self.hidden = [None for _ in range(8)]
        self.last_pfoot = None
        self.last_tran = None
        self.floor_y = []
        self.temp_hidden = None
        self.first_reach = True

    @staticmethod
    def cat(*x):
        """
        Concatenate the tensors.
        """
        return [torch.cat(_, dim=1) for _ in zip(*x)]

    @torch.no_grad()
    def forward_online(self, j2dc, accc, oric, first_tran=None, first_frame=False):
        r"""
        Forward for one frame.
        :param j2dc: Tensor [33, 3].
        :param accc: Tensor [imu_num, 3].
        :param oric: Tensor [imu_num, 3, 3].
        :return: Pose [24, 3, 3] and tran [3].
        """

        def cat(*x):
            return torch.cat([_.flatten(0) for _ in x]).view(1, -1)

        def f(i, x):  # rnn_i(x)
            rnn = self.__getattr__('rnn%d' % (i + 1))
            x, self.hidden[i] = rnn.rnn(relu(rnn.linear1(x), inplace=True).unsqueeze(1), self.hidden[i])
            return rnn.linear2(x.squeeze(1))

        def fk(glb_pose):
            glb_pose = glb_pose.view(1, 24, 3, 3)
            pb = torch.stack([glb_pose[:, body_model.parent[i]].matmul(self.b[i]) for i in range(1, 24)], dim=1)
            pb = torch.cat((torch.zeros(glb_pose.shape[0], 1, 3, device=device), pb.squeeze(-1)), dim=1)
            return body_model.bone_vector_to_joint_position(pb)[0]

        # visual confidence
        c = j2dc[:, -1].mean().item()
        Rcr = oric.view(Net.imu_num, 3, 3)[-1]

        # inertial
        accr = accc.clone().view(Net.imu_num, 3).mm(Rcr)
        orir = Rcr.t().matmul(oric.clone().view(Net.imu_num, 3, 3))
        j3dr_i = f(1, cat(accr, orir))
        vr = f(2, cat(accr, orir, j3dr_i))

        # visual
        j2dc_clone = j2dc.clone().to(j2dc.device)
        if c > self.conf_range[0] or first_frame:
            j2dc_clone[:, :2] = j2dc_clone[:, :2] / (get_bbox_scale(j2dc_clone))
            j2dc_clone[24:, :2] = j2dc_clone[24:, :2] - j2dc_clone[23:24, :2]
            j2dc_clone[:23, :2] = j2dc_clone[:23, :2] - j2dc_clone[23:24, :2]
            j3dc = f(3, cat(accc, oric, j2dc_clone))
            j3dr_v = j3dc.view(23, 3).mm(Rcr)
            if first_frame:
                pc = f(5, cat(accc, oric, j2dc, j3dc))

        # lerp
        if c >= self.conf_range[1]:
            j3dr = j3dr_v
            pc = f(5, cat(accc, oric, j2dc, j3dc))
        elif c > self.conf_range[0]:
            k = (c - self.conf_range[0]) / (self.conf_range[1] - self.conf_range[0])
            j3dr = art.math.lerp(j3dr_i.view(-1), j3dr_v.view(-1), k)
            pc = f(5, cat(accc, oric, j2dc, j3dc))
        else:
            j3dr = j3dr_i

        poseg6d = f(6, cat(accr, orir, j3dr))
        contact = f(7, cat(accr, orir, j3dr)).sigmoid()

        # pose
        poseg = art.math.r6d_to_rotation_matrix(poseg6d).view(1, 24, 3, 3)
        pose = body_model.inverse_kinematics_R(poseg)[0]
        pose[0] = Rcr

        # update inertial hidden states
        if c >= self.conf_range[1] and self.use_imu_updater and self.first_reach:
            self.first_reach = False
            nd, nh = 2, self.hidden_size
            rnn = self.__getattr__('rnn2')
            h_init, c_init = rnn.init_net(j3dr.flatten()).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
            self.hidden[1] = (h_init, c_init)

        # tran
        pfoot = fk(poseg)[10:12].mm(Rcr.t())
        if contact.max() < self.contact_threshold or self.last_pfoot is None:
            v = Rcr.mm(vr.view(3, 1)).view(3) * vel_scale / 60
        else:
            v = (self.last_pfoot - pfoot)[contact.argmax().item()]
        if self.last_tran is None:
            tran = v
        else:
            tran = self.last_tran.to(device) + v

        if self.conf_range[1] <= c:
            k = (c - self.conf_range[0]) / (self.conf_range[1] - self.conf_range[0])
            if k > 1:
                k = 1
            if (pc - tran).norm() > self.distrance_threshold or self.tran_filter_num > 1:
                tran = pc
            else:
                tran = art.math.lerp(tran, pc, self.tran_filter_num * k)

        # determine floor height
        tran = tran.view(3)
        self.gravityc = self.gravityc.to(device)
        if len(self.floor_y) < 11 and not first_frame and first_tran is None and contact.max() > self.contact_threshold and self.use_flat_floor and c >= self.conf_range[1]:
            p0 = torch.dot(pfoot[0] + tran.to(device), self.gravityc) * self.gravityc
            p1 = torch.dot(pfoot[1] + tran.to(device), self.gravityc) * self.gravityc
            if p0.norm() < p1.norm():
                self.floor_y.append(p1)
            else:
                self.floor_y.append(p0)
        if self.use_flat_floor and len(self.floor_y) > 10 and contact.max() > self.contact_threshold:
            p0 = torch.dot(pfoot[0] + tran.to(device), self.gravityc) * self.gravityc
            p1 = torch.dot(pfoot[1] + tran.to(device), self.gravityc) * self.gravityc
            if p0.norm() < p1.norm() and (sum(self.floor_y[-6:]) / 6 - p1).norm() < self.height_threhold:
                tran = tran + (sum(self.floor_y[-6:]) / 6 - p1)
            elif (sum(self.floor_y[-6:]) / 6 - p0).norm() < self.height_threhold:
                tran = tran + (sum(self.floor_y[-6:]) / 6 - p0)
        if first_tran is not None:
            tran = first_tran.to(device)
        elif first_frame:
            tran = pc.to(device)

        self.last_pfoot = pfoot
        if self.use_reproj_opt or self.use_vision_updater:
            if not self.live:
                grot, joint, vert = body_model.forward_kinematics(pose.view(-1, 24, 3, 3), tran=tran.view(-1, 3).to(device),
                                                              calc_mesh=True)
                j = sync_mp3d(vert[0], joint[0])
            else:
                if self.update_vision_count == 0:
                    grot, joint, vert = body_model.forward_kinematics(pose.view(-1, 24, 3, 3), tran=tran.view(-1, 3).to(device),
                                                                      calc_mesh=True)
                    j = sync_mp3d(vert[0], joint[0])
                    self.j_temp = j
                    self.update_vision_count = self.update_vision_freq
                else:
                    j = self.j_temp
                    self.update_vision_count -= 1

        # optimize reprojection error
        if self.use_reproj_opt and c > self.conf_range[0]:
            p = j2dc[:, 2]
            # optimize x, y
            ax = (p / j[:, 2].pow(2)).sum() + self.smooth
            bx = (p * (- j[:, 0] / j[:, 2].pow(2) + j2dc[:, 0] / j[:, 2])).sum()
            ay = (p / j[:, 2].pow(2)).sum() + self.smooth
            by = (p * (- j[:, 1] / j[:, 2].pow(2) + j2dc[:, 1] / j[:, 2])).sum()
            d_tran = torch.tensor([bx / ax, by / ay, 0]).to(device)
            tran = tran + d_tran
            # optimize z
            j = j + d_tran
            az = (p * (j[:, 0].pow(2) + j[:, 1].pow(2)) / j[:, 2].pow(4)).sum() + self.smooth
            bz = (p * ((j[:, 0] / j[:, 2] - j2dc[:, 0]) * j[:, 0] / j[:, 2].pow(2) + (
                    j[:, 1] / j[:, 2] - j2dc[:, 1]) * j[:, 1] / j[:, 2].pow(2))).sum()
            d_tran = torch.tensor([0, 0, bz / az]).to(device)
            tran = tran + d_tran
            j = j + d_tran.to(device)

        # visual forward for hidden states
        if self.use_vision_updater and c <= self.conf_range[0] and (self.update_vision_count == self.update_vision_freq or not self.live):
            j2dc = j / j[:, 2:]
            j3dc = joint[0][1:] - joint[0][:1]
            f(5, cat(accc, oric, j2dc, j3dc))
            j2dc[:, :2] = j2dc[:, :2] / get_bbox_scale(j2dc)
            j2dc[24:, :2] = j2dc[24:, :2] - j2dc[23:24, :2]
            j2dc[:23, :2] = j2dc[:23, :2] - j2dc[23:24, :2]
            f(3, cat(accc, oric, j2dc)).view(23, 3)

        self.last_tran = tran
        return pose.view(24, 3, 3).cpu(), tran.view(3).cpu()


def get_bbox_scale(uv):
    """
    Get the scale of the bbox of the uv coordinates.
    max(bbox width, bbox height).
    """
    u_max, u_min = uv[..., 0].max(dim=-1).values, uv[..., 0].min(dim=-1).values
    v_max, v_min = uv[..., 1].max(dim=-1).values, uv[..., 1].min(dim=-1).values
    return torch.max(u_max - u_min, v_max - v_min)


def sync_mp3d(vert, joint):
    """
    Sync the mediapipe 3d vertices with the SMPL joints and vertices.
    :param vert:
    :param joint:
    :return:
    """
    syn_3d = vert[mp_mask]
    syn_3d[11:17] = joint[16:22].clone()
    syn_3d[23:25] = joint[1:3].clone()
    syn_3d[25:27] = joint[4:6].clone()
    syn_3d[27:29] = joint[7:9].clone()
    return syn_3d
