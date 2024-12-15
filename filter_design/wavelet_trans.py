import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_radon import Radon


def haar_wavelet_transform(image):
    """
    哈里小波变换函数
    参数：
        image (torch.Tensor): 输入图像，尺寸为 (batch_size, channels, height, width)
    返回：
        tuple: 四个分量 (LL, HH, HL, LH)，每个分量的尺寸为 (batch_size, channels, height//2, width//2)
    """
    device = image.device
    batch_size, channels, height, width = image.shape
    conv_ll = nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False).to(device)
    conv_ll.weight = nn.Parameter(torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device).unsqueeze(0).unsqueeze(0).repeat(channels, channels, 1, 1), requires_grad=False)
    conv_lh = nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False).to(device)
    conv_lh.weight = nn.Parameter(torch.tensor([[0.5, -0.5], [0.5, -0.5]], device=device).unsqueeze(0).unsqueeze(0).repeat(channels, channels, 1, 1), requires_grad=False)
    conv_hl = nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False).to(device)
    conv_hl.weight = nn.Parameter(torch.tensor([[0.5, 0.5], [-0.5, -0.5]], device=device).unsqueeze(0).unsqueeze(0).repeat(channels, channels, 1, 1), requires_grad=False)
    conv_hh = nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False).to(device)
    conv_hh.weight = nn.Parameter(torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=device).unsqueeze(0).unsqueeze(0).repeat(channels, channels, 1, 1), requires_grad=False)
    LL, LH, HL, HH = conv_ll(image), conv_lh(image), conv_hl(image), conv_hh(image)
    return LL, HH, HL, LH


def haar_wavelet_inverse_transform(LL, LH, HL, HH):
    """
    Perform 2D Haar wavelet inverse transform on a batch of images.
    Parameters:
    - LL (torch.Tensor): Low-frequency component, shape (b, c, h/2, w/2)
    - LH (torch.Tensor): Horizontal high-frequency component, shape (b, c, h/2, w/2)
    - HL (torch.Tensor): Vertical high-frequency component, shape (b, c, h/2, w/2)
    - HH (torch.Tensor): Diagonal high-frequency component, shape (b, c, h/2, w/2)
    Returns:
    - reconstructed (torch.Tensor): Reconstructed images, shape (b, c, h, w)
    """
    bs, ch, h_half, w_half = LL.shape
    # 初始化重构图像
    reconstructed = torch.zeros(bs, ch, h_half * 2, w_half * 2, device=LL.device)
    a, b, c, d = (LL + LH + HL + HH)*0.5, (LL - LH + HL - HH)*0.5, (LL + LH - HL - HH)*0.5, (LL - LH - HL + HH)*0.5
    reconstructed[:, :, 0::2, 0::2] = a
    reconstructed[:, :, 0::2, 1::2] = b
    reconstructed[:, :, 1::2, 0::2] = c
    reconstructed[:, :, 1::2, 1::2] = d
    return reconstructed


class WaveletFilterNet(nn.Module):
    def __init__(self, h, w, alpha=1):
        super(WaveletFilterNet, self).__init__()
        t_shape = (1, 1, h // 2, w // 2)
        # self.filter = torch.cat([(alpha-0.1) * torch.ones(t_shape), alpha*torch.ones(t_shape), (alpha+0.1)*torch.ones(t_shape)], dim=1)
        self.filter = torch.cat([
            (alpha - 0.1) * torch.ones(t_shape, device='cuda'),  # 张量 1
            alpha * torch.ones(t_shape, device='cuda'),  # 张量 2
            (alpha + 0.1) * torch.ones(t_shape, device='cuda')  # 张量 3
        ], dim=1)  # 在 dim=1（第二个维度）拼接
        self.filter = nn.Parameter(self.filter, requires_grad=True)

    def forward(self, x):
        # 进行二维离散小波变换
        b, c, h, w = x.shape
        l_x, LH, HL, HH = haar_wavelet_transform(x)
        h_x = torch.cat((LH, HL, HH), dim=1)  # shape: (b, 3c, h/2, w/2)
        del LH, HL, HH
        filtered_hx = h_x * self.filter.repeat(b, 1, 1, 1)
        filtered_lh, filtered_hl, filtered_hh = torch.chunk(filtered_hx, 3, dim=1)
        recon_x = haar_wavelet_inverse_transform(l_x, filtered_lh, filtered_hl, filtered_hh)

        return recon_x


if __name__ == '__main__':
    # 测试代码
    device = 'cuda:7'
    # img = torch.rand(8, 8)  # 创建一个示例图像（8x8）
    hd_pic = np.load('/home/ssd/linshuijin/PETrecon_backup/simulation_angular/angular_180/test_transverse_picHD.npy', allow_pickle=True)[0:2, :, :]
    ndct_sino = np.load('/home/ssd/linshuijin/PETrecon_backup/simulation_angular/angular_180/test_transverse_sinoLD.npy', allow_pickle=True)[0:2, :, :]

    img = torch.from_numpy(ndct_sino).float().to(device).unsqueeze(1)
    # img = torch.cat([img, img.flip(dims=[-2])], dim=-1)
    # plt.imshow(img.squeeze().detach().cpu().numpy()[0], 'gray')
    # plt.show()
    img1 = img[:, :, :, 0:90]
    angles_1 = np.linspace(0, 180, 180, endpoint=False)[0:90]
    radon = Radon(resolution=128, angles=np.deg2rad(angles_1), clip_to_circle=True)
    filtered_sinogram = radon.filter_sinogram(img1.transpose(-1, -2), filter_name='ramp')
    # print(time.time() - t_3)
    fbp = radon.backprojection(filtered_sinogram)
    print("重构图像:")
    plt.imshow(fbp.squeeze().detach().cpu().numpy()[0], 'gray')
    plt.show()

    diff = np.abs(hd_pic[0, 0, :, :] - fbp.squeeze().detach().cpu().numpy()[0])

    # 创建一个空的RGB图像
    rgb_img = np.zeros((diff.shape[0], diff.shape[1], 3))

    # 映射到白色和红色之间的颜色
    # 使用自定义的颜色映射：'Reds' 是Matplotlib的红色渐变色
    plt.imshow(diff, cmap='Reds', vmin=0, vmax=1)  # 将灰度图映射到白色和红色之间
    plt.axis('off')
    plt.show()

    img2 = img[:, :, :, 1::2]
    angles_2 = np.linspace(1, 180, 180, endpoint=False)[0::2]
    radon = Radon(resolution=128, angles=np.deg2rad(angles_2), clip_to_circle=True)
    filtered_sinogram = radon.filter_sinogram(img2.transpose(-1, -2), filter_name='ramp')
    # print(time.time() - t_3)
    fbp1 = radon.backprojection(filtered_sinogram)
    print("重构图像:")
    plt.imshow(fbp1.squeeze().detach().cpu().numpy()[0], 'gray')
    plt.show()

    diff1 = np.abs(hd_pic[0, 0, :, :] - fbp1.squeeze().detach().cpu().numpy()[0])

    # 映射到白色和红色之间的颜色
    # 使用自定义的颜色映射：'Reds' 是Matplotlib的红色渐变色
    plt.imshow(diff1, cmap='Reds', vmin=0, vmax=1)  # 将灰度图映射到白色和红色之间
    plt.axis('off')
    plt.show()



    theta_s, theta_e, num_theta = 0, 360, 360
    pic = torch.from_numpy(ld_pic).float().to(device)
    angles_1 = np.linspace(theta_s, theta_e, num_theta, endpoint=False)
    radon = Radon(resolution=128, angles=np.deg2rad(angles_1), clip_to_circle=True)
    recon_sino_360 = radon.forward(pic)
    plt.imshow(recon_sino_360.squeeze().detach().cpu().numpy()[0], 'gray')
    plt.show()


    img = img[:, :, :, 120:300]
    # net = WaveletFilterNet(128, 180).to(device)
    # recon = net(img)
    theta_s, theta_e = 120, 300
    angles_1 = np.linspace(theta_s, theta_e, 180, endpoint=False)
    angles_2 = np.linspace(0, np.pi, 180, endpoint=False)
    angles_3 = np.linspace(0, np.pi, 60, endpoint=False)
    radon = Radon(resolution=128, angles=np.deg2rad(angles_1), clip_to_circle=True)
    filtered_sinogram = radon.filter_sinogram(img.transpose(-1, -2), filter_name='ramp')
    # print(time.time() - t_3)
    fbp = radon.backprojection(filtered_sinogram)
    print("重构图像:")
    plt.imshow(fbp.squeeze().detach().cpu().numpy()[0], 'gray')
    plt.show()
    del net, recon, filtered_sinogram, fbp

    img = radon.filter_sinogram(img.transpose(-1, -2))
    img = radon.backprojection(img)

    # LL, LH, HL, HH = haar_wavelet_transform(img)
    # recon = haar_wavelet_inverse_transform(LL, LH, HL, HH)
    print("原始图像:")
    plt.imshow(img.squeeze().detach().cpu().numpy()[0], 'gray')
    plt.show()


    print("低频部分 (LL):")
    plt.imshow(LL.squeeze()[0], 'gray')
    plt.show()
    print("水平方向高频部分 (LH):")
    # print(LH)
    plt.imshow(LH.squeeze()[0], 'gray')
    plt.show()
    print("垂直方向高频部分 (HL):")
    print(HL)
    print("对角线高频部分 (HH):")
    print(HH)