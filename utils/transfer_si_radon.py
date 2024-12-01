import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F


def iradon_torch(sinogram, theta=None, output_size=None,
                 filter_name="ramp", interpolation="linear", circle=True,
                 preserve_range=True):
    """
    Inverse radon transform using PyTorch, with support for CUDA and gradients.

    Parameters
    ----------
    sinogram : torch.Tensor
        Sinogram with shape (batch_size, channels, rows, angles).
    theta : torch.Tensor, optional
        Angles for the projections, should have the same number of elements as angles dimension in sinogram.
        If not provided, angles will be evenly spaced between 0 and 180 degrees.
    output_size : int, optional
        Size of the output reconstruction (output will be square).
    filter_name : str, optional
        Filter used in frequency domain filtering, default is "ramp".
    interpolation : str, optional
        Interpolation method, default is "linear". Supports 'linear', 'nearest', or 'cubic'.
    circle : bool, optional
        If True, the output will be masked to an inscribed circle.
    preserve_range : bool, optional
        Whether to keep the original range of values.

    Returns
    -------
    reconstructed : torch.Tensor
        Reconstructed images, with shape (batch_size, channels, output_size, output_size).
    """

    device = sinogram.device  # Get the device (CPU or GPU)
    b, c, rows, angles = sinogram.shape

    if theta is None:
        theta = torch.linspace(0., 180., angles, device=device, dtype=sinogram.dtype, requires_grad=False)

    # Ensure theta has the correct shape
    assert theta.shape[0] == angles, "Theta must match the number of angles."

    # Fourier padding
    projection_size_padded = max(64, int(2 ** torch.ceil(torch.log2(torch.tensor(rows * 2, dtype=torch.float32)))))
    sinogram_padded = F.pad(sinogram, (0, 0, 0, projection_size_padded - rows), mode='constant', value=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter_torch(projection_size_padded, filter_name, device=device)
    sinogram_fft = torch.fft.fft(sinogram_padded, dim=2)
    sinogram_filtered = torch.fft.ifft(sinogram_fft * fourier_filter, dim=2)
    sinogram_filtered = sinogram_filtered[:, :, :rows, :].real  # Trim padding and keep real part

    if output_size is None:
        output_size = rows if circle else int(torch.floor(torch.sqrt(torch.tensor(rows ** 2 / 2.0))))

    reconstructed = torch.zeros((b, c, output_size, output_size), dtype=sinogram.dtype, device=device)
    radius = output_size // 2
    xpr, ypr = torch.meshgrid(torch.arange(output_size, device=device) - radius,
                              torch.arange(output_size, device=device) - radius, indexing='ij')

    for i in range(angles):
        angle = torch.deg2rad(theta[i])
        t = ypr * torch.cos(angle) - xpr * torch.sin(angle)
        t = t.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)

        # Interpolation
        if interpolation == 'linear':
            sinogram_filtered_reshaped = sinogram_filtered.permute(0, 1, 3, 2)  # Shape (b, c, angles, rows)
            reconstructed += F.grid_sample(sinogram_filtered_reshaped, t.unsqueeze(-1), mode='bilinear',
                                           padding_mode='zeros', align_corners=True)
        elif interpolation == 'nearest':
            reconstructed += F.grid_sample(sinogram_filtered_reshaped, t.unsqueeze(-1), mode='nearest',
                                           padding_mode='zeros', align_corners=True)
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation}")

    # Apply circular mask if needed
    if circle:
        mask = xpr ** 2 + ypr ** 2 > radius ** 2
        reconstructed[:, :, mask] = 0

    return reconstructed * torch.pi / (2 * angles)



# 假设 radon_transform_torch 是你已经实现的拉东变换函数
def radon_transform_torch(img, n_theta, device='cuda'):
    """
    :param img: 输入的图像，形状为 (b, c, h, w)，要求已经在GPU上。
    :param theta: 要投影的角度序列，单位是弧度，默认是从0到180度的投影。
    :param device: 使用的设备，默认为'cuda'。
    :return: sinogram，形状为 (b, c, r, theta)。
    """
    b, c, h, w = img.shape

    theta = torch.linspace(0, math.pi, n_theta, device=device)

    # 计算对角线长度，作为投影后的长度r
    diag_len = h
    pad_h = (diag_len - h) // 2
    pad_w = (diag_len - w) // 2

    # 为了确保投影长度足够，需要在图像边缘填充
    img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h))
    sinogram = torch.zeros((b, c, diag_len, len(theta)), device=device)

    # 生成归一化的grid
    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, diag_len, device=device),
                                      torch.linspace(-1, 1, diag_len, device=device)), -1)

    for i, angle in enumerate(theta):
        # 创建旋转矩阵
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)

        # 旋转grid
        rot_grid = torch.matmul(grid.view(-1, 2), rotation_matrix).view(diag_len, diag_len, 2)

        # 添加批次和通道维度，扩展为4D
        rot_grid = rot_grid.unsqueeze(0).repeat(b, 1, 1, 1)

        # 使用 grid_sample 进行插值
        rotated_img = F.grid_sample(img_padded, rot_grid, align_corners=True)

        # 对旋转后的图像进行积分（即求和）
        sinogram[:, :, :, i] = rotated_img.sum(dim=-1)

    return sinogram


# 1. 生成测试图像 (Shepp-Logan Phantom)
image = shepp_logan_phantom()
image = resize(image, (128, 128))  # 缩放到128x128大小
image_torch = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

# 2. 使用 skimage 计算 Radon 变换
theta = np.linspace(0., 180., 180, endpoint=False)
skimage_sinogram = radon(image, theta=theta, circle=True)

# 3. 使用 PyTorch 计算 Radon 变换
# theta_torch = torch.tensor(np.deg2rad(theta), dtype=torch.float32).cuda()
torch_sinogram = radon_transform_torch(image_torch, 180).cpu().squeeze().numpy()

# 4. 可视化原始图像、skimage的正弦图、PyTorch的正弦图和差异
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# 原始图像
ax[0, 0].set_title("Original Image")
ax[0, 0].imshow(image, cmap='gray')

# skimage的正弦图
ax[0, 1].set_title("Skimage Radon Transform")
ax[0, 1].imshow(skimage_sinogram, cmap='gray', aspect='auto')

# PyTorch的正弦图
ax[1, 0].set_title("PyTorch Radon Transform")
ax[1, 0].imshow(torch_sinogram, cmap='gray', aspect='auto')

# 差异图
ax[1, 1].set_title("Difference (Torch - Skimage)")
diff = np.abs(torch_sinogram - skimage_sinogram)
ax[1, 1].imshow(diff, cmap='hot', aspect='auto')

plt.tight_layout()
plt.show()
