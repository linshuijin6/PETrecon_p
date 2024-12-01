import os
import torch
import numpy
import numpy as np
import scipy.io as sio
from utils.transfer_si import i2s
import matplotlib.pyplot as plt
from utils.radon import Radon

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# for UDPET-brain
path_root = "/mnt/data/linshuijin/data_PET/data_UDPET_brain"
path_sep = 'train'
file_names = os.listdir(os.path.join(path_root, path_sep))
file_pre_list = {}
for i, file_names in enumerate(file_names):
    file_pre = file_names.split('.')[0].split('_')
    file_pre_n = ''.join([k+'_' for k in file_pre[:-1]])
    # file_pre_n = file_pre[0] + '_' + file_pre[1] + '_' + file_pre[2] + '_' + file_pre[3] + '_' + file_pre[4] + '_'
    if file_pre_n not in file_pre_list.keys():
        file_pre_list[file_pre_n] = 0
    else:
        file_pre_list[file_pre_n] = file_pre_list[file_pre_n] + 1

file_pre_l = []
for f in file_pre_list.keys():
    if file_pre_list[f] > 0:
        file_pre_l.append(f)

file_names = []
for file_pre in file_pre_l:
    for i in range(40):
        file_name_whole = file_pre + str(i+40) + '.mat'
        file_names.append(os.path.join(path_sep, file_name_whole))

path_sep = 'test'
file_path_names = os.listdir(os.path.join(path_root, path_sep))
file_pre_list1 = {}
for i, file_name in enumerate(file_path_names):
    file_pre = file_name.split('_')
    file_pre_n = file_pre[0] + '_' + file_pre[1] + '_' + file_pre[2] + '_' + file_pre[3] + '_' + file_pre[4] + '_'
    if file_pre_n not in file_pre_list1.keys():
        file_pre_list1[file_pre_n] = 0
    else:
        file_pre_list1[file_pre_n] = file_pre_list1[file_pre_n] + 1

file_pre_l1 = []
for f in file_pre_list1.keys():
    if file_pre_list1[f] > 0:
        file_pre_l1.append(f)

for file_pre in file_pre_l1:
    for i in range(40):
        file_name_whole = file_pre + str(i+40) + '.mat'
        file_names.append(os.path.join(path_sep, file_name_whole))

ldSinos = []
hdSinos = []
ldImgs = []
hdImgs = []
k = 0
n = 50000
# temPath = '/media/linshuijin/LENOVO_USB_HDD/project_pet/PETrecon/tmp_180_128128/'
# geoMatrix = []
# geoMatrix.append(np.load(temPath + 'geoMatrix-0.npy', allow_pickle=True))
for i, file_name in enumerate(file_names):
    file_path = os.path.join(path_root, file_name)
    try:
        raw_data = sio.loadmat(file_path, mat_dtype=True)['img']
    except ValueError as e:
        print(f"Error loading {file_path}: {e}")
        print(file_name)
        continue
    # raw_data = sio.loadmat(file_path)['img']
    ldImg = raw_data[:, 0:128, :]
    hdImg = raw_data[:, 128:256, :]
    # ldImgs.append(ldImg), hdImgs.append(hdImg)
    # torch_ldImgs = torch.from_numpy(np.array(ldImg)).to('cuda').unsqueeze(1)
    # dSinos = i2s(torch_ldImgs, 0, sinogram_nAngular=180, geoMatrix=geoMatrix)
    if n*k <= i < n*(k+1):
        ldImgs.append(ldImg), hdImgs.append(hdImg)
        # print(i)
    if i > n*(k+1):
        break

torch_ldImgs = torch.from_numpy(np.array(ldImgs)).to('cuda').float()
torch_hdImgs = torch.from_numpy(np.array(hdImgs)).to('cuda').float()
me_radon = Radon(circle=True)
ldSinos = me_radon(torch_ldImgs)
hdSinos = me_radon(torch_hdImgs)
# ldSinos = i2s(torch_ldImgs, 0, sinogram_nAngular=180, geoMatrix=geoMatrix)
ldSino_np = ldSinos.cpu().numpy()
hdSino_np = hdSinos.cpu().numpy()
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_sinoLD.npy', ldSino_np)


# hdSinos = i2s(torch_hdImgs, 0, sinogram_nAngular=180, geoMatrix=geoMatrix)
# hdSino_np = hdSinos.cpu().numpy()
np.save(f'/mnt/data/linshuijin/PETrecon_p/simulation_angular/angular_180/transverse_sinoHD.npy', hdSino_np)
np.save(f'/mnt/data/linshuijin/PETrecon_p/simulation_angular/angular_180/transverse_sinoLD.npy', ldSino_np)
np.save(f'/mnt/data/linshuijin/PETrecon_p/simulation_angular/angular_180/transverse_picLD.npy', ldImgs)
np.save(f'/mnt/data/linshuijin/PETrecon_p/simulation_angular/angular_180/transverse_picHD.npy', hdImgs)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_sinoLD.npy', ldSino_np)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_sinoHD.npy', hdSino_np)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_picLD.npy', ldImgs)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_picHD.npy', hdImgs)
