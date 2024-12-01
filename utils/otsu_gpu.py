import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def gaussian_blur_2d(input, kernel_size=5, sigma=1.0):
    """
    对4D张量进行高斯模糊（GPU加速）。

    参数:
        input (torch.Tensor): 输入图像张量，尺寸为 (batch_size, channels, height, width)。
        kernel_size (int): 高斯核大小，需为奇数。
        sigma (float): 高斯标准差。

    返回:
        torch.Tensor: 模糊后的图像张量。
    """

    # 生成高斯核
    def gaussian_kernel_2d(kernel_size, sigma):
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
        x = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = x / x.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d

    kernel_2d = gaussian_kernel_2d(kernel_size, sigma).to(input.device)
    kernel_2d = kernel_2d.expand(input.shape[1], 1, kernel_size, kernel_size)  # 扩展为多通道

    # 使用 F.conv2d 进行卷积操作
    padding = kernel_size // 2
    blurred = F.conv2d(input, kernel_2d, padding=padding, groups=input.shape[1])

    return blurred


def otsu_threshold_batch(img_batch):
    # Step 1: 计算每个图像的直方图（在灰度值范围0-255）
    batch_size, channels, height, width = img_batch.shape
    hist = torch.stack([torch.histc(img_batch[i], bins=256, min=0, max=255) for i in range(batch_size)], dim=0)  # shape: (batch_size, 256)
    hist = hist / hist.sum(dim=1, keepdim=True)  # 归一化直方图，每个图像的灰度分布

    # Step 2: 计算累积和和累积均值
    cumsum_hist = torch.cumsum(hist, dim=1)  # 累积和，shape: (batch_size, 256)
    cumsum_mean = torch.cumsum(hist * torch.arange(256, device=img_batch.device), dim=1)  # 累积均值，shape: (batch_size, 256)
    global_mean = cumsum_mean[:, -1]  # 全局均值，shape: (batch_size,)

    # Step 3: 计算类间方差
    numerator = (global_mean.unsqueeze(1) * cumsum_hist - cumsum_mean) ** 2
    denominator = cumsum_hist * (1 - cumsum_hist)
    between_class_variance = numerator / (denominator + 1e-6)  # 避免除零

    # Step 4: 获取最大方差对应的阈值
    _, optimal_thresholds = torch.max(between_class_variance, dim=1)  # shape: (batch_size,)

    # Step 5: 根据最优阈值生成掩膜
    optimal_thresholds = optimal_thresholds.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)  # 调整阈值形状
    mask_batch = (img_batch >= optimal_thresholds).float()  # 将掩膜转换为0和1的浮点型结果

    return mask_batch


def morphological_closing(binary_img, kernel_size):
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=binary_img.device)
    kernel = kernel.expand(binary_img.shape[1], 1, kernel_size, kernel_size)

    # 先进行腐蚀，再进行膨胀
    dilated = F.conv2d(1 - binary_img, kernel, padding=kernel_size // 2, groups=binary_img.shape[1])
    dilated = (dilated < 1).float()
    closed = F.conv2d(dilated, kernel, padding=kernel_size // 2, groups=binary_img.shape[1])
    closed = (closed > 0).float()

    return closed


# 示例用法
img = np.load('/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_sinoHD.npy', allow_pickle=True)
img_torch = torch.from_numpy(img).to('cuda').unsqueeze(1).float()  # 假设图像在[0, 1]范围内
min_val = img_torch.amin(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)
max_val = img_torch.amax(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)

img_normalized = 255*(img_torch - min_val) / (max_val - min_val + 1e-6)  # 避免除零
# img_torch = torch.norm(img_torch, p=2, dim=1)  # 转换为单通道灰度图
mask_torch = otsu_threshold_batch(img_normalized.to(torch.uint8))
mask_torch1 = gaussian_blur_2d(mask_torch, kernel_size=13, sigma=2.0)
# mask_torch1 = morphological_closing(mask_torch, kernel_size=7)
min_val = mask_torch1.amin(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)
max_val = mask_torch1.amax(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)

mask_torch1 = 255*(mask_torch1 - min_val) / (max_val - min_val + 1e-6)
mask_torch2 = otsu_threshold_batch(mask_torch1.to(torch.uint8))

img_normalized = img_normalized*mask_torch
plt.imshow(mask_torch[2, 0, :, :].cpu().numpy(), cmap='gray'), plt.show()
plt.imshow(mask_torch1[2, 0, :, :].cpu().numpy(), cmap='gray'), plt.show()
plt.imshow(mask_torch2[2, 0, :, :].cpu().numpy(), cmap='gray'), plt.show()
plt.imshow(img_normalized[2, 0, :, :].cpu().numpy()), plt.show()
plt.imshow(img_torch[2, 0, :, :].cpu().numpy()), plt.show()


