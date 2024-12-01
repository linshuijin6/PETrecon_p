import matplotlib.pyplot as plt
import torch
import numpy as np

from utils import Radon
from utils.transfer_si import i2s


def set_random_pixels_to_zero(data, ratio):
    """
    将输入张量中的部分像素值随机设为0.

    参数:
        data (torch.Tensor): 输入张量，形状为 (batchsize, c, h, w).
        ratio (float): 要设为0的像素占比（0到1之间的浮点数）。

    返回:
        torch.Tensor: 修改后的张量.
    """
    # 确保 ratio 在 [0, 1] 范围内
    if not (0 <= ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")

    batchsize, c, h, w = data.shape
    # 计算要设为0的像素数量
    num_pixels_to_zero = int(batchsize * c * h * w * ratio)

    # 随机选择要设为0的索引
    indices = torch.randperm(batchsize * c * h * w)[:num_pixels_to_zero]

    # 将对应的像素值设为0
    data_flat = data.view(-1)  # 展平张量
    data_flat[indices] = 0
    return data_flat.view(batchsize, c, h, w)  # 还原原始形状


# 示例
# data = torch.randn(2, 3, 128, 128)  # 生成一个随机张量
device = torch.device('cpu')
data = np.load('/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_picHD.npy', allow_pickle=True)
data = torch.from_numpy(data).to(device).float()
tem = data.clone()
# data = data[:, None, :, :]
plt.imshow(data[0, 0, :, :].cpu().numpy(), cmap='gray'), plt.show()
ratio = 0.5  # 设为0的像素占比
modified_data = set_random_pixels_to_zero(data, ratio)
plt.imshow(modified_data[0, 0, :, :].numpy(), cmap='gray'), plt.show()

torch_noise_ldImgs = modified_data.to(device)
torch_ldImgs = tem.to(device)

radon = Radon(180, circle=True, device=data.device)
noise_sino = radon(torch_noise_ldImgs)
sino = radon(torch_ldImgs)
# temPath = '/mnt/data/linshuijin/PETrecon/tmp_180_128*128/'
# geoMatrix = []
# geoMatrix.append(np.load(temPath + 'geoMatrix-0.npy', allow_pickle=True))
# noise_sino = i2s(torch_noise_ldImgs, geoMatrix, 180)
plt.imshow(noise_sino[0, 0, :, :]), plt.title('noise')
plt.show()

# sino = i2s(torch_ldImgs, geoMatrix, 180)
plt.imshow(sino[0, 0, :, :]), plt.title('origin')
plt.show()

1

