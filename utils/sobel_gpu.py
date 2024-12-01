import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def sobel(img_torch):
    # img_batch = (img_torch - min_val) / (max_val - min_val + 1e-6)  # 避免除零
    batch_size, channels, height, width = img_torch.shape
    # img_batch = torch.randn(batch_size, channels, height, width).to('cuda')  # 示例数据，假设在[-1, 1]范围

    # Step 1: 定义 Sobel 卷积核
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32, device=img_torch.device).view(1, 1, 3, 3)

    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=torch.float32, device=img_torch.device).view(1, 1, 3, 3)

    # 将 Sobel 卷积核扩展到 (channels, channels, kernel_size, kernel_size) 维度
    sobel_kernel_x = sobel_kernel_x.expand(channels, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_y.expand(channels, 1, 3, 3)

    # Step 2: 应用 Sobel 卷积核来计算边缘强度
    grad_x = F.conv2d(img_torch, sobel_kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(img_torch, sobel_kernel_y, padding=1, groups=channels)

    # 计算梯度幅值
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # shape: (batch_size, channels, height, width)

    # Step 3: 使用阈值生成二值掩膜
    # 这里我们可以设定一个经验阈值，也可以使用 Otsu 方法之类的自适应阈值
    threshold = 0.5  # 可根据图像特征调整阈值
    mask_batch = (edge_magnitude > threshold).float()  # 二值化掩膜
    return mask_batch


if __name__ == "__main__":
    # 假设我们有一个形状为 (batch_size, channels, height, width) 的图像张量
    img = np.load('../simulation_angular/angular_180/test_transverse_sinoHD.npy',
                  allow_pickle=True)
    img_torch = torch.from_numpy(img).to('cuda').unsqueeze(1).float()  # 假设图像在[0, 1]范围内
    min_val = img_torch.amin(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)
    max_val = img_torch.amax(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)
    img_torch = (img_torch - min_val) / (max_val - min_val + 1e-6)  # 避免除零

    mask_batch = sobel(img_torch)
    # 输出掩膜形状以验证
    print(mask_batch.shape)  # 应为 (batch_size, channels, height, width)

    img_normalized = img_torch * mask_batch
    plt.imshow(mask_batch[2, 0, :, :].cpu().numpy(), cmap='gray'), plt.show()
    plt.imshow(img_normalized[2, 0, :, :].cpu().numpy()), plt.show()
    plt.imshow(img_torch[2, 0, :, :].cpu().numpy()), plt.show()
