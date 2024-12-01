import numpy as np
import torch
from matplotlib import pyplot as plt

# from radon import Radon


def normalization2one(input_tensor: torch.Tensor or np.ndarray) -> torch.Tensor:
    # 假设输入的 tensor 形状为 (batchsize, channels=1, h, w) 或 numpy 形状为 (batchsize, h, w)
    # input_tensor = torch.randn(4, 1, 64, 64)  # 示例的输入张量
    input_tensor = torch.from_numpy(input_tensor).to('cuda').unsqueeze(1).float() if isinstance(input_tensor, np.ndarray) else input_tensor

    # 为了每个 batch 归一化，我们要按batch维度进行最小值和最大值的计算
    # 计算每个batch的最小值和最大值，保持维度为 (batchsize, 1, 1, 1)
    min_val = input_tensor.reshape(input_tensor.size(0), -1).min(dim=1)[0].reshape(-1, 1, 1, 1)
    max_val = input_tensor.reshape(input_tensor.size(0), -1).max(dim=1)[0].reshape(-1, 1, 1, 1)

    # 进行归一化，将所有数值归一化到 [0, 1] 区间
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + 1e-8)  # 1e-8 防止除以0
    assert input_tensor.shape == normalized_tensor.shape
    # normalized_tensor = normalized_tensor * 1.0 - 0.5

    return normalized_tensor  # 确认输出形状 (batchsize, 1, h, w)


def min_max_normalized(data=torch.rand(2, 3, 4, 4)):
    assert len(data.shape) == 4, "Input data should be 4D tensor"

    # 最大最小归一化
    data_min = torch.min(torch.min(torch.min(data, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0], dim=0, keepdim=True)[0]
    data_max = torch.max(torch.max(torch.max(data, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0], dim=0, keepdim=True)[0]

    min_max_normalized = (data - data_min) / (data_max - data_min + 1e-8)  # 防止除以零
    # min_max_normalized = min_max_normalized * 1.0 - 0.5
    return min_max_normalized


def rms_normalized(data=torch.rand(2, 3, 4, 4)):
    assert len(data.shape) == 4, "Input data should be 4D tensor"

    # 均方根归一化
    data_rms = torch.sqrt(torch.mean(data ** 2, dim=(2, 3), keepdim=True))
    rms_normalized = data / (data_rms + 1e-8)  # 防止除以零
    return rms_normalized


def log_normalized(data=torch.rand(2, 3, 4, 4)):
    assert len(data.shape) == 4, "Input data should be 4D tensor"

    # 对数归一化
    log_normalized = torch.log(data + 1e-8)  # 防止对数值为零
    return log_normalized


def plot_data(data, title, order, color_me, xlim):
    data = data.detach() if data.requires_grad else data
    data = data.cpu() if data.is_cuda else data
    data = data.numpy() if isinstance(data, torch.Tensor) else data
    data = data.reshape(-1)
    plt.subplot(2, 2, order)
    plt.hist(data, bins=30, color=color_me[order-1], alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlim(xlim)
    plt.grid(axis='y', alpha=0.75)


def plot_pic(data, title, order):
    data = data.detach() if data.requires_grad else data
    data = data.cpu() if data.is_cuda else data
    data = data.numpy() if isinstance(data, torch.Tensor) else data
    plt.subplot(2, 2, order)
    plt.imshow(data[0, 0, :, :])
    plt.title(title)
    plt.colorbar()
    plt.grid(False)

if __name__ == "__main__":
    import torch

    # 示例数据 (batch_size, channels, height, width)
    # data = torch.rand(2, 3, 4, 4)  # 2个样本，3个通道，4x4图像
    data_n = np.load('../simulation_angular/angular_180/test_transverse_sinoHD.npy', allow_pickle=True)
    xlim = (data_n.reshape(-1).min()-1, data_n.reshape(-1).max()+1)
    data = torch.from_numpy(data_n).to('cuda').float().unsqueeze(1)

    # 最大最小归一化
    min_max_normalized = normalization2one(data)

    # 均方根归一化
    rms_normalized = rms_normalized(data)

    # 对数归一化
    log_normalized = log_normalized(data)
    color_me = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    # 打印结果
    # 绘制频数分布直方图
    plt.figure(figsize=(12, 8))
    plot_data(data, "Original Data", 1, color_me, xlim)
    plot_data(min_max_normalized, "Min-Max Normalized", 2, color_me, xlim)
    # plot_data(rms_normalized, "RMS Normalized", 3, color_me, xlim)
    # plot_data(log_normalized, "Log Normalized", 4, color_me, xlim)
    plt.show()

    # 绘制图像
    plt.figure(figsize=(12, 8))
    plot_pic(data, "Original Data", 1)
    plot_pic(min_max_normalized, "Min-Max Normalized", 2)
    plot_pic(rms_normalized, "RMS Normalized", 3)
    plot_pic(log_normalized, "Log Normalized", 4)
    plt.show()

    radon_me = Radon(n_theta=180, circle=True, device='cuda')
    pic_recon_list = []

    plt.figure(figsize=(12, 8))
    for i, data_sino in enumerate((data, min_max_normalized, rms_normalized, log_normalized)):
        pic_recon_t = radon_me.filter_backprojection(data_sino)
        plot_pic(pic_recon_t, f"Reconstructed Pic{i+1}", i+1)
    plt.show()


    # print("最大最小归一化结果:\n", min_max_normalized)
    # print("均方根归一化结果:\n", rms_normalized)
    # print("对数归一化结果:\n", log_normalized)

