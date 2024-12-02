import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.data import normalization2one


def show_images(imgs, titles=None, keep_range=True, shape=None, figsize=(8, 8.5)):
    # imgs: eg, [x, sinogram, filtered_sinogram, fbp], and x.device=='cuda'，torch，shape:h, w
    # titles = ["Original Image", "Sinogram", "Filtered Sinogram", "Filtered Backprojection"]
    imgs = [x.cpu().numpy() for x in imgs]

    # Get the min and max of all images
    if keep_range:
        combined_data = np.array(imgs)
        _min, _max = np.amin(combined_data), np.amax(combined_data)
    else:
        _min, _max = None, None

    if titles is None:
        titles = [str(i) for i in range(combined_data.shape[0])]

    if shape is None:
        shape = (1, len(imgs))

    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax[i].imshow(img, cmap=plt.cm.Greys_r, vmin=_min, vmax=_max)
        ax[i].set_title(title)


def visual_plot(pics, titles, args, psnr_avg, ssim_avg, show=True):
    # 定义图像数据和标题
    # pics = [x1, x2, x1_denoised, aver_x, picLD, mid_recon, s_in_1, s_in_2, s_out_1, s_out_2,
    #         i_out_m12, i_in_m2, i_out_m21, i_in_m1, aver_x, sino_label, mid_recon, Y]
    # titles = ["x1", "x2", "x1_denoised", "aver_x", "input_LD", "mid_recon", "sino_p1", "sino_p2", "sino_recon_p1",
    #           "sino_recon_p2",
    #           'i_out_m12', 'i_in_m2', 'i_out_m21', 'i_in_m1', 'aver_recon_sino', 'sino_label', 'pic_recon', 'pic_label']
    assert len(pics) % 2 == 0, "The number of pictures should be even"
    assert len(pics) == len(titles), "The number of pictures and titles should be the same"
    n_rows = len(pics) // 2
    # 设置图像整体大小
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 18))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.3, wspace=0.2)  # 减少图像间的间距

    # 绘制每个子图
    for i, (pic, title) in enumerate(zip(pics, titles)):
        ax = axes[i // 2, i % 2]  # 计算当前的行列位置
        pic_t = pic[0, 0, :, :].cpu().detach().numpy() if len(pic.shape) == 4 else pic[0, :, :].numpy()
        ax.imshow(pic_t)  # 使用灰度显示图像
        ax.set_title(title, fontsize=10)  # 设置子图标题=
        # 以下代码组合移除坐标轴、刻度和边框
        ax.axis('off')  # 移除边框和坐标轴
        ax.get_xaxis().set_ticks([])  # 移除 x 轴刻度
        ax.get_yaxis().set_ticks([])  # 移除 y 轴刻度
        ax.spines['top'].set_visible(False)  # 隐藏顶部边框
        ax.spines['bottom'].set_visible(False)  # 隐藏底部边框
        ax.spines['left'].set_visible(False)  # 隐藏左侧边框
        ax.spines['right'].set_visible(False)  # 隐藏右侧边框
        # 设置大标题
        if psnr_avg:
            fig.suptitle(
                f'psnr: {psnr_avg}, ssim: {ssim_avg}', fontsize=14)
        plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    # 调整布局，避免重叠
    if args:
        os.mkdir(os.path.join(args.log_dir, 'pic_visual')) if not os.path.exists(
            os.path.join(args.log_dir, 'pic_visual')) else None
        pic_path = os.path.join(args.log_dir, 'pic_visual', f'test_psnr_{psnr_avg}_ssim_{ssim_avg}.png')
        plt.savefig(pic_path, format='png', dpi=300, bbox_inches='tight')  # 300 dpi保证高分辨率
    if show:
        plt.show()


# 计算PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(recon, label, max_val=1.0):
    mse = F.mse_loss(recon, label, reduction='mean')  # Mean Squared Error (均方误差)
    psnr = 10 * torch.log10(max_val ** 2 / mse)  # 计算 PSNR
    return psnr


# 计算SSIM (Structural Similarity Index Measure)
def calculate_ssim(recon, label, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    # 创建高斯窗口
    def gaussian_window(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= (window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
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


def calculate_metrics(recon: torch.Tensor, label) -> tuple:
    # recon的尺寸为 (batchsize, h, w)， numpy格式
    recon, label = normalization2one(recon).cpu().numpy().squeeze(), normalization2one(
        label.cpu().numpy()).cpu().numpy().squeeze()  # 归一化到 [0, 1] 区间
    bs, h, w = recon.shape
    psnr_scores = np.array([psnr(recon[i], label[i], data_range=1.0) for i in range(bs)])
    ssim_scores = np.array([ssim(recon[i], label[i], data_range=1.0) for i in range(bs)])
    # average_psnr = np.mean(psnr_scores)
    # average_ssim = np.mean(ssim_scores)
    return psnr_scores, ssim_scores


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
