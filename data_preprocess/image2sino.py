import os
import torch
import numpy
import numpy as np
import scipy.io as sio

from utils import Radon
from utils.transfer_si import i2s
import matplotlib.pyplot as plt
# from skimage.transform import iradon, radon, warp


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# for UDPET-brain
path_root = "/home/ssddata/linshuijin/data/UDPET_Brain/train_mat"
file_names = os.listdir(path_root)
ldSinos = []
hdSinos = []
ldImgs = []
hdImgs = []
k = 0
n = 10000000
# temPath = '/mnt/data/linshuijin/PETrecon/tmp_180_128*128/'
# geoMatrix = []
# geoMatrix.append(np.load(temPath + 'geoMatrix-0.npy', allow_pickle=True))
for i, file_name in enumerate(file_names):
    file_path = os.path.join(path_root, file_name)
    raw_data = sio.loadmat(file_path)['img']
    ldImg = raw_data[:, 0:128, :]
    ldImg = np.rot90(ldImg, -1, (1, 2))
    hdImg = raw_data[:, 128:256, :]
    hdImg = np.rot90(hdImg, -1, (1, 2))
    # plt.imshow(ldImg[0, :, :]), plt.title('LD')
    # plt.show()
    # plt.imshow(hdImg[0, :, :]), plt.title('HD')
    # plt.show()
    # ldImgs.append(ldImg), hdImgs.append(hdImg)
    torch_ldImgs = torch.from_numpy(np.array(ldImg)).to('cuda').squeeze(1)
    # dSinos = i2s(torch_ldImgs, 0, sinogram_nAngular=180, geoMatrix=geoMatrix)
    if n*k <= i < n*(k+1):
        ldImgs.append(ldImg), hdImgs.append(hdImg)
    if i >= n*(k+1):
        break

torch_ldImgs = torch.from_numpy(np.array(ldImgs)).to('cuda').float()
torch_hdImgs = torch.from_numpy(np.array(hdImgs)).to('cuda').float()

ldimg_np = torch_ldImgs.cpu().numpy()
hdimg_np = torch_hdImgs.cpu().numpy()

radon_me = Radon(n_theta=180, circle=True, device='cuda')
ldSinos = radon_me(torch_ldImgs)
hdSinos = radon_me(torch_hdImgs)
#
# ldSinos = i2s(torch_ldImgs, geoMatrix=geoMatrix, sinogram_nAngular=180, counts=1.4e5)
# hdSinos = i2s(torch_hdImgs, geoMatrix=geoMatrix, sinogram_nAngular=180, counts=2.2e5)

ldSino_np = ldSinos.squeeze(1).cpu().numpy()
hdSino_np = hdSinos.squeeze(1).cpu().numpy()
# plt.imshow(ldSino_np[2, 0, :, :]), plt.show()
# plt.imshow(hdSino_np[2, 0, :, :]), plt.show()
# plt.imshow(ldimg_np[2, 0, :, :]), plt.show()
# plt.imshow(hdimg_np[2, 0, :, :]), plt.show()

# recon_ld = radon_me.filter_backprojection(ldSinos)
# recon_hd = radon_me.filter_backprojection(hdSinos)
# plt.imshow(recon_ld[0, 0, :, :].cpu().numpy()), plt.show()
# plt.imshow(recon_hd[0, 0, :, :].cpu().numpy()), plt.show()
# diff_sino = abs(ldSino_np - hdSino_np)
#
#
# truesFraction=1  #  0~1， 数值越大，有效信号越多
# randomsFraction=0.1  #  0~1, 数值越大，随机散射越多
# ldSinos_noise = i2s(torch_ldImgs, geoMatrix=geoMatrix, sinogram_nAngular=180, counts=1e7,
#                     randomsFraction=randomsFraction)
# ldSino_noise_np = ldSinos_noise.cpu().numpy()
# plt.imshow(ldSino_noise_np[0, :, :], 'gray'), plt.show()
#
# differ = ldSino_noise_np - ldSino_np
# plt.imshow(differ[0, :, :], 'coolwarm'), plt.title(f't{str(truesFraction)}_r{str(randomsFraction)}')
# cmap = plt.get_cmap('coolwarm')  # 选择颜色映射
# norm = plt.Normalize(vmin=-max(abs(differ[0, :, :].reshape(-1))), vmax=max(abs(differ[0, :, :].reshape(-1))))
#
# plt.imshow(diff_sino[0, :, :], 'coolwarm'), plt.title('differ_sino')
# # 添加颜色条
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.set_label('Difference Values')
# plt.show()
#
# cmap = plt.get_cmap('coolwarm')  # 选择颜色映射
# # norm = plt.Normalize(vmin=-max(abs(differ[0, :, :].reshape(-1))), vmax=max(abs(differ[0, :, :].reshape(-1))))
#
# # 添加颜色条
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm)
# cbar.set_label('Difference Values')
# plt.show()
#
#
# randomsFraction_list=[0.9, 0.7, 0.5, 0.3, 0.1]
# # fig, axes = plt.subplots(5, 2, figsize=(12, 18))
# for i, randomsFraction in enumerate(randomsFraction_list):
#     # ax = axes[i%2, i//2]
#     ldSinos_noise2 = i2s(torch_ldImgs, geoMatrix=geoMatrix, sinogram_nAngular=180, counts=1e7,
#                          randomsFraction=randomsFraction)
#     ldSino_noise2_np = ldSinos_noise2.cpu().numpy()
#     plt.imshow(ldSino_noise2_np[0, :, :]), plt.title(f'r={randomsFraction}'), plt.show()
#     differ2 = abs(ldSino_noise2_np - ldSino_np)
#     # bx = axes[(i+1)%2, (i+1)//2]
#     plt.imshow(differ2[0, :, :], 'coolwarm'), plt.title(f't{str(truesFraction)}_r{str(randomsFraction)}')
#     # 添加颜色条
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm)
#     plt.show()
#
#     pic_recon = iradon(ldSino_noise2_np[0, :, :], theta=np.linspace(0, 180, 180), circle=True)
#     plt.imshow(pic_recon), plt.title(f'noise, r={randomsFraction}'), plt.show()
#
#     pic_recon2 = iradon(ldSino_np[0, :, :], theta=np.linspace(0, 180, 180), circle=True)
#     plt.imshow(pic_recon2), plt.title(f'clean, r={randomsFraction}'), plt.show()
#
#     diff_recon = abs(pic_recon - pic_recon2)
#     plt.imshow(diff_recon), plt.title(f'diff, r={randomsFraction}'), plt.show()
#
# # np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_sinoLD.npy', ldSino_np)
# # del ldSinos, ldSino_np, torch_ldImgs
#
# hdSinos = i2s(torch_hdImgs, geoMatrix=geoMatrix, sinogram_nAngular=180, counts=1e9)


# hdSino_np = hdSinos.cpu().numpy()
# plt.imshow(hdSino_np[0, :, :]), plt.show()
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_sinoHD.npy', hdSino_np)
np.save(f'/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/transverse_sinoLD.npy', ldSino_np)
np.save(f'/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/transverse_sinoHD.npy', hdSino_np)
np.save(f'/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/transverse_picHD.npy', hdimg_np)
np.save(f'/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/transverse_picLD.npy', ldimg_np)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_sinoLD.npy', ldSino_np)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_sinoHD.npy', hdSino_np)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_picLD.npy', ldImgs)
# np.save(f'/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_p{k+1}_4_picHD.npy', hdImgs)
