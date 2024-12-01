import torch
import torch.nn as nn


def create_high_pass_filter(c, h, w, cutoff_frequency):
    """
    创建一个高通滤波器
    """
    # 均匀网格，坐标值从 -h/2 到 h/2, -w/2 到 w/2
    x = torch.linspace(-w // 2, w // 2, w)
    y = torch.linspace(-h // 2, h // 2, h)

    # 生成网格坐标
    X, Y = torch.meshgrid(x, y)

    # 计算每个点到中心的距离
    distance = torch.sqrt(X ** 2 + Y ** 2)

    # 创建高通滤波器：低频部分为0，高频部分为1
    # 低频部分（距离中心较小的部分）值设为0，距离较远的部分则为1
    high_pass_filter = (distance > cutoff_frequency).float()

    # 扩展到多通道 (c) 和大小 (h, w)
    return high_pass_filter.view(1, 1, h, w).expand(c, -1, -1, -1)


class HighPassFilter(nn.Module):
    def __init__(self, c, h, w, cutoff_frequency=0.1):
        super(HighPassFilter, self).__init__()

        # 初始化高通滤波器的参数：设置为 (c, h, w)
        self.cutoff_frequency = cutoff_frequency
        self.c = c
        self.h = h
        self.w = w

        # 创建高通滤波器（低频部分为0，高频部分为1）
        self.filter = nn.Parameter(create_high_pass_filter(c, h, w, cutoff_frequency))

    def forward(self, x):
        """
        对输入图像进行傅里叶变换、滤波后返回结果
        """
        # 进行傅里叶变换
        X = torch.fft.fft2(x)

        # 应用高通滤波器
        filtered_X = X * self.filter

        # 反傅里叶变换，得到滤波后的图像
        x_filtered = torch.fft.ifft2(filtered_X)

        return x_filtered


if __name__ == "__main__":
    # 假设输入数据 x 的形状为 (batch_size, c, h, w)
    batch_size = 4
    c = 3  # 通道数，例如 RGB 图像
    h = 32  # 高度
    w = 32  # 宽度
    x = torch.randn(batch_size, c, h, w)  # 随机生成输入数据

    # 创建 HighPassFilter 实例
    model = HighPassFilter(c, h, w, cutoff_frequency=5)

    # 进行前向传播
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
