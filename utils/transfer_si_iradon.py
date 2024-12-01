import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon
from skimage.transform import radon
import torch.nn.functional as F
from skimage.transform import resize


def _sinogram_circle_to_square(sinogram, device='cuda'):
    import torch.nn.functional
    # 计算对角线长度（向上取整）
    diagonal = int(torch.ceil(torch.sqrt(torch.tensor(2.0, device=device)) * sinogram.shape[2]))

    # 计算需要填充的大小
    pad = diagonal - sinogram.shape[2]
    old_center = sinogram.shape[2] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center

    # 定义填充的宽度，先在上下方向进行填充
    pad_width = (pad_before, pad - pad_before)

    # 使用 torch.nn.functional.pad 进行填充
    sinogram_padded = torch.nn.functional.pad(sinogram, (0, 0, pad_width[0], pad_width[1]), mode='constant', value=0)

    return sinogram_padded



def iradon_torch(sinogram, n_theta=None, output_size=None,
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
    _, _, output_size, angles = sinogram.shape
    theta = torch.linspace(0., n_theta, angles, device=device, dtype=sinogram.dtype, requires_grad=False)


    # Ensure theta has the correct shape
    assert theta.shape[0] == angles, "Theta must match the number of angles."
    sinogram = _sinogram_circle_to_square(sinogram)

    b, c, rows, _ = sinogram.shape

    # Fourier padding
    projection_size_padded = max(64, int(2 ** torch.ceil(torch.log2(torch.tensor(rows * 2, dtype=torch.float32)))))
    sinogram_padded = F.pad(sinogram, (0, 0, 0, projection_size_padded - rows), mode='constant', value=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter_torch(projection_size_padded, filter_name, device=device)
    fourier_filter = fourier_filter[None, None, :, :]
    sinogram_fft = torch.fft.fft(sinogram_padded, dim=2)
    sinogram_filtered = torch.fft.ifft(sinogram_fft * fourier_filter, dim=2)
    sinogram_filtered = sinogram_filtered[:, :, :rows, :].real.float()  # Trim padding and keep real part


    reconstructed = torch.zeros((b, c, output_size, output_size), dtype=sinogram.dtype, device=device)
    radius = output_size // 2
    xpr, ypr = torch.meshgrid(torch.arange(output_size, device=device) - radius,
                              torch.arange(output_size, device=device) - radius)

    for i in range(angles):
        angle = torch.deg2rad(theta[i])
        cos_a = torch.cos(angle).float()
        sin_a = torch.sin(angle).float()
        t = ypr * cos_a - xpr * sin_a
        t = t.unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
        # 生成仿射矩阵 R
        R = torch.tensor([[cos_a, -sin_a, 0],
                          [sin_a, cos_a, 0]], dtype=torch.float32)
        grid = F.affine_grid(R.unsqueeze(0), reconstructed.size()).to(device)

        # Interpolation
        if interpolation == 'linear':
            sinogram_filtered_reshaped = sinogram_filtered.permute(0, 1, 3, 2)  # Shape (b, c, angles, rows)
            reconstructed += F.grid_sample(sinogram_filtered_reshaped, grid, mode='bilinear',
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


def _get_fourier_filter_torch(size, filter_name, device):
    """ Generate the desired Fourier filter in PyTorch. """
    # 使用 torch.arange 生成两个张量
    part1 = torch.arange(1, size // 2 + 1, 2, dtype=torch.int)
    part2 = torch.arange(size // 2 - 1, 0, -2, dtype=torch.int)

    # 使用 torch.cat 进行拼接
    n = torch.cat((part1, part2))
    f = torch.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    omega = 2 * torch.fft.fft(f)


    if filter_name == "ramp":
        return torch.real(omega).unsqueeze(-1).to(device)
    elif filter_name == "shepp-logan":
        return torch.abs(omega) * torch.sinc(omega / (2 * torch.pi))
    elif filter_name == "cosine":
        return torch.abs(omega) * torch.cos(omega / 2)
    elif filter_name == "hamming":
        return torch.abs(omega) * (0.54 + 0.46 * torch.cos(omega))
    elif filter_name == "hann":
        return torch.abs(omega) * (1 + torch.cos(omega)) / 2
    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def iradon_transform_torch(sino, n_theta, device='cuda'):
    b, c, r, theta_size = sino.shape
    theta = torch.linspace(0, np.pi, n_theta, device=device)

    image = torch.zeros((b, c, r, r), device=device)
    center = (r - 1) / 2

    for i in range(theta_size):
        angle = theta[i]
        sinogram_slice = sino[:, :, :, i].repeat(1, r, 1)
        x = torch.arange(0, r, device=device) - center
        y = torch.arange(0, r, device=device) - center
        x, y = torch.meshgrid(x, y)

        x_theta = x * torch.cos(angle) + y * torch.sin(angle)
        x_theta = (x_theta + (r - 1) / 2).long().clamp(0, r - 1)

        # 进行索引聚合，形状转换
        # x_theta的形状是 (h, w)，需要扩展为 (b, c, h, w) 以进行正确的聚合
        sinogram_values = sinogram_slice.gather(1, x_theta.unsqueeze(0))  # 获取sinogram对应的值
        image += sinogram_values.sum(dim=2)  # 聚合到重建图像中

    image /= theta_size
    return image


# 生成示例数据
# b, c, r, theta_size = 1, 1, 182, 180
# # 使用skimage生成sinogram作为参考
# x = np.zeros((182, 182))
# x[50:130, 50:130] = 1  # 创建一个简单的正方形图像
# theta = np.linspace(0., np.pi, theta_size, endpoint=False)
# sino = radon(x, theta=theta, circle=True)
# sino_torch = torch.tensor(sino[np.newaxis, np.newaxis, :, :], device='cuda')
#
# 1. 生成测试图像 (Shepp-Logan Phantom)
image = shepp_logan_phantom()
image = resize(image, (128, 128))  # 缩放到128x128大小
image_torch = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

# 2. 使用 skimage 计算 Radon 变换
theta = np.linspace(0., 180., 180, endpoint=False)
sino = radon(image, theta=theta, circle=True)
sino_torch = torch.tensor(sino[np.newaxis, np.newaxis, :, :], device='cuda')

# 使用skimage的iradon进行参考
image_reference = 1

# 使用自定义的IRadon实现
image_custom = iradon_torch(sino_torch, 180)


# 转换为numpy以便可视化
image_custom_np = image_custom.squeeze().cpu().detach().numpy()
image_reference_np = image_reference


# 可视化结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Reconstructed Image (Custom IRadon)')
plt.imshow(image_custom_np, cmap='gray')
plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.title('Reconstructed Image (skimage IRadon)')
# plt.imshow(image_reference_np, cmap='gray')
# plt.axis('off')

plt.tight_layout()
plt.show()
