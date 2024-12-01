import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def sobel(img):
    # img_batch = img
    batch_size, channels, height, width = img.shape
    # img_batch = torch.randn(batch_size, channels, height, width).to('cuda')  # 示例数据，假设在[-1, 1]范围

    # Step 1: 定义 Sobel 卷积核
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    # 将 Sobel 卷积核扩展到 (channels, channels, kernel_size, kernel_size) 维度
    sobel_kernel_x = sobel_kernel_x.expand(channels, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_y.expand(channels, 1, 3, 3)

    # Step 2: 应用 Sobel 卷积核来计算边缘强度
    grad_x = F.conv2d(img, sobel_kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(img, sobel_kernel_y, padding=1, groups=channels)

    # 计算梯度幅值
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # shape: (batch_size, channels, height, width)

    # Step 3: 使用阈值生成二值掩膜
    # 这里我们可以设定一个经验阈值，也可以使用 Otsu 方法之类的自适应阈值
    threshold = 0.5  # 可根据图像特征调整阈值
    mask_batch = (edge_magnitude > threshold).float()  # 二值化掩膜
    return mask_batch

# 假设输入图像张量 (batch_size, channels, height, width)
batch_size, channels, height, width = 8, 1, 128, 180
img_batch1 = torch.randn(batch_size, channels, height, width).to('cuda')  # 示例数据
# 假设我们有一个形状为 (batch_size, channels, height, width) 的图像张量
img = np.load('/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_sinoHD.npy',
              allow_pickle=True)
img_torch = torch.from_numpy(img).to('cuda').unsqueeze(1).float()  # 假设图像在[0, 1]范围内
min_val = img_torch.amin(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)
max_val = img_torch.amax(dim=(2, 3), keepdim=True)  # shape: (batch_size, channels, 1, 1)
img_torch = (img_torch - min_val) / (max_val - min_val + 1e-6)  # 避免除零
# 先将图像二值化（例如通过大津法，或设定固定阈值），这里用固定阈值进行示例
# threshold = 0.5
binary_img_batch = sobel(img_torch)  # 生成二值图像

# 定义卷积核，用于形态学操作（3x3的结构元素，用于膨胀或腐蚀）
# 例如使用一个 3x3 全 1 的卷积核
kernel = torch.ones((1, 1, 5, 5), device=img_torch.device)
 # 扩展 kernel 以支持批次和通道数
kernel = kernel.expand(channels, 1, 5, 5)  # (channels, 1, 3, 3)

# 形态学膨胀操作（max pooling 的方式）
def morphological_dilation(binary_img_batch, kernel):
    # 进行二维卷积，用最大池化来实现膨胀
    dilated = F.conv2d(binary_img_batch, kernel, padding=2, groups=channels)
    dilated = (dilated > 0).float()  # 将膨胀后的值转换回二值化
    return dilated

# 形态学腐蚀操作（min pooling 的方式）
def morphological_erosion(binary_img_batch, kernel):
    # 进行二维卷积，用最小池化来实现腐蚀
    eroded = F.conv2d(1 - binary_img_batch, kernel, padding=2, groups=channels)
    eroded = (eroded < 1).float()  # 将腐蚀后的值转换回二值化
    return eroded

# 执行膨胀和腐蚀操作
dilated_batch = morphological_dilation(binary_img_batch, kernel)
eroded_batch = morphological_erosion(binary_img_batch, kernel)

img_normalized = img_torch * eroded_batch
plt.imshow(dilated_batch[2, 0, :, :].cpu().numpy(), cmap='gray'), plt.show()
plt.imshow(eroded_batch[2, 0, :, :].cpu().numpy(), cmap='gray'), plt.show()
plt.imshow(img_normalized[2, 0, :, :].cpu().numpy()), plt.show()
plt.imshow(img_torch[2, 0, :, :].cpu().numpy()), plt.show()
# 输出结果形状以验证
print("膨胀结果形状:", dilated_batch.shape)  # 应为 (batch_size, channels, height, width)
print("腐蚀结果形状:", eroded_batch.shape)  # 应为 (batch_size, channels, height, width)
