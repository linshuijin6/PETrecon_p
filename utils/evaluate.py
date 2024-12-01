from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from normalize import normalization2one


# 计算PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(recon, label, max_val=1.0):
    mse = F.mse_loss(recon, label, reduction='mean')  # Mean Squared Error (均方误差)
    psnr = 10 * torch.log10(max_val**2 / mse)         # 计算 PSNR
    return psnr

# 计算SSIM (Structural Similarity Index Measure)
def calculate_ssim(recon, label, window_size=11, C1=0.01**2, C2=0.03**2):
    # 创建高斯窗口
    def gaussian_window(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= (window_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g.reshape(1, 1, window_size)

    # 定义窗口
    window = gaussian_window(window_size)
    window = window.expand(recon.size(1), 1, window_size).to(recon.device)

    # 计算各类统计量
    mu1 = F.conv2d(recon, window, padding=window_size // 2, stride=1, groups=recon.size(1))
    mu2 = F.conv2d(label, window, padding=window_size // 2, stride=1, groups=label.size(1))

    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(recon * recon, window, padding=window_size // 2, groups=recon.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(label * label, window, padding=window_size // 2, groups=label.size(1)) - mu2_sq
    sigma12 = F.conv2d(recon * label, window, padding=window_size // 2, groups=recon.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()  # 返回均值SSIM

# 计算其他潜在指标 (扩展性示例)
# 可以根据需要加入其他指标计算函数

# 主函数，用于批量处理recon和label
def evaluate_metrics(recon, label, metrics=['psnr', 'ssim']):
    results = {'psnr': 0, 'ssim': 0}
    if 'psnr' in metrics:
        results['psnr'] += psnr(recon, label)
    if 'ssim' in metrics:
        results['ssim'] += ssim(recon, label)
    # 添加其他指标计算
    return results


def calculate_metrics(recon: Union[np.array], label) -> tuple:
    assert len(recon.shape) == 3 and len(label.shape) == 3, "输入图像的维度应为 (batchsize, h, w)"
    recon, label = normalization2one(recon).cpu().numpy().squeeze(), normalization2one(label).cpu().numpy().squeeze()  # 归一化到 [0, 1] 区间
    bs, h, w = recon.shape
    psnr_scores = np.array([psnr(recon[i], label[i], data_range=1.0) for i in range(bs)])
    ssim_scores = np.array([ssim(recon[i], label[i], data_range=1.0) for i in range(bs)])
    average_psnr = np.mean(psnr_scores)
    average_ssim = np.mean(ssim_scores)
    return average_psnr, average_ssim


def plot_box(data_list, title_list, ylabel, xlabel, save_path=None, show=True):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_list, labels=title_list, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"))
    plt.ylabel(ylabel)
    plt.title(xlabel)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


if __name__ == "__main__":
    # 示例：计算重建图像和标签图像的 PSNR 和 SSIM
    recon = np.random.rand(4, 64, 64)  # 重建图像
    label = np.random.rand(4, 64, 64)  # 标签图像
    psnr_average, ssim_average = calculate_metrics(recon, label)  # 计算 PSNR 和 SSIM
    data = list(np.random.rand(5, 100))
    labels = ['OSEM', 'MAPEM', 'DeepPET', 'FBSEM', 'Proposed']
    plot_box(data, labels, 'PSNR (dB)', '(a) PSNR', save_path='./boxplot.png', show=True)

    print(psnr_average, ssim_average)
