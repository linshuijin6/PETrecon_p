import numpy as np
import torch
from skimage.transform import iradon
import matplotlib.pyplot as plt
from utils.transfer_si import s2i, i2s

AN = np.load('simulation_angular/angular_180/sino_AN.npy', allow_pickle=True)
pic = np.load('simulation_angular/angular_180/pic_HD.npy', allow_pickle=True)
# 创建正弦图
# theta = np.linspace(0., 360., max(sinogram.shape), endpoint=False)

sino_me = i2s_batch(torch.from_numpy(pic), AN)


# 使用滤波反投影 (FBP) 重建
# reconstruction_fbp = iradon(sinogram[2, :, :], theta=theta, filter_name='ramp')
#
# sinogram = torch.from_numpy(sinogram)
# recon_s2i = s2i(sinogram[2, :, :]).cpu()
# 显示重建图像
# plt.imshow(reconstruction_fbp, cmap='gray')
# plt.show()

plt.imshow(sino_me[2, :, :], cmap='gray')
plt.show()

plt.imshow(pic[2, :, :], cmap='gray')
plt.show()
