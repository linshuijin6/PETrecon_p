import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon
from skimage.transform import radon
import torch.nn.functional as F
from skimage.transform import resize

from radon import Radon


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
    b, c, rows, angles = sinogram.shape

    theta = torch.linspace(0., n_theta, angles, device=device, dtype=sinogram.dtype, requires_grad=False)

    # Ensure theta has the correct shape
    assert theta.shape[0] == angles, "Theta must match the number of angles."

    # Fourier padding
    projection_size_padded = max(64, int(2 ** torch.ceil(torch.log2(torch.tensor(rows * 2, dtype=torch.float32)))))
    sinogram_padded = F.pad(sinogram, (0, 0, 0, projection_size_padded - rows), mode='constant', value=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter_torch(projection_size_padded, filter_name, device=device)
    fourier_filter = fourier_filter[:, None, :, None]
    sinogram_fft = torch.fft.fft(sinogram_padded, dim=2)
    sinogram_filtered = torch.fft.ifft(sinogram_fft * fourier_filter, dim=2)
    sinogram_filtered = sinogram_filtered[:, :, :rows, :].real.float()  # Trim padding and keep real part

    if output_size is None:
        output_size = rows if circle else int(torch.floor(torch.sqrt(torch.tensor(rows ** 2 / 2.0))))

    reconstructed = torch.zeros((b, c, output_size, output_size), dtype=sinogram.dtype, device=device)
    radius = output_size // 2
    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, output_size, device=device),
                                      torch.linspace(-1, 1, output_size, device=device)), -1)
    # xpr, ypr = torch.meshgrid(torch.arange(output_size, device=device) - radius, torch.arange(output_size, device=device) - radius)

    for i in range(angles):
        angle = torch.deg2rad(theta[i])
        cos_a = torch.cos(angle).float()
        sin_a = torch.sin(angle).float()
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device)
        # 旋转grid
        rot_grid = torch.matmul(grid.view(-1, 2), rotation_matrix).view(output_size, output_size, 2)
        t = rot_grid.unsqueeze(0).repeat(b, 1, 1, 1)

        # Interpolation
        if interpolation == 'linear':
            sinogram_filtered_reshaped = sinogram_filtered.permute(0, 1, 3, 2)  # Shape (b, c, angles, rows)
            reconstructed += F.grid_sample(sinogram_filtered_reshaped, t, mode='bilinear',
                                           padding_mode='zeros', align_corners=True)
        elif interpolation == 'nearest':
            reconstructed += F.grid_sample(sinogram_filtered_reshaped, t, mode='nearest',
                                           padding_mode='zeros', align_corners=True)
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation}")

    # Apply circular mask if needed
    # if circle:
    #     mask = xpr ** 2 + ypr ** 2 > radius ** 2
    #     reconstructed[:, :, mask] = 0

    return reconstructed * torch.pi / (2 * angles)


def _get_fourier_filter_torch(size, filter_name, device):
    """ Generate the desired Fourier filter in PyTorch. """
    f = torch.fft.fftfreq(size, device=device).unsqueeze(0)
    omega = 2 * torch.pi * f

    if filter_name == "ramp":
        return torch.abs(omega)
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
image_reference = iradon(sino.squeeze(), theta=theta, circle=True)

# 使用自定义的IRadon实现
radon_c = Radon(circle=True, device='cuda')
image_custom = radon_c.filter_backprojection(sino_torch)
# sino_c = radon_c.forward(image_torch)
# image_custom = 1


# 转换为numpy以便可视化
image_custom_np = image_custom.detach().cpu().squeeze().numpy()
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

plt.subplot(1, 3, 3)
plt.title('Reconstructed Image (skimage IRadon)')
plt.imshow(image_reference_np, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
