import torch
import os
import glob

input_dir = r'D:\yxy\Xsens_DOT_Data_Exporter-2021.0.0-win\data\20230124_021319'
data = {}
for file in glob.glob(os.path.join(input_dir, '*.csv')):
    with open(file, 'r') as f:
        sep = f.readline()[-2]
        print('sep:', sep)
        header = f.readline().split(sep)
        qw = header.index('Quat_W')
        qx = header.index('Quat_X')
        qy = header.index('Quat_Y')
        qz = header.index('Quat_Z')
        ax = header.index('Acc_X')
        ay = header.index('Acc_Y')
        az = header.index('Acc_Z')

        quats, accs = [], []
        for line in f.readlines():
            l = line.split(sep)
            quats.append([float(l[qw]), float(l[qx]), float(l[qy]), float(l[qz])])
            accs.append([float(l[ax]), float(l[ay]), float(l[az])])

        data[os.path.basename(file).split('_')[1]] = {'q': torch.tensor(quats), 'a': torch.tensor(accs)}

torch.save(data, os.path.join(input_dir, 'data.pt'))
print('Save at', os.path.join(input_dir, 'data.pt'))
