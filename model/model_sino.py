import numpy as np
import torch
import torch.nn as nn
import math
from matplotlib import pyplot as plt


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.height = height
        self.width = width
        self.dim = dim

        # 创建位置编码
        pos_encoding = torch.arange(0, d_model, 2)
        if dim % 2 != 0:
            raise ValueError("Embedding dimension must be even for sinusoidal encoding.")

        # 生成编码
        d_model = dim // 2  # 一半用于行编码，另一半用于列编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 行编码
        pos_w = torch.arange(0, width).unsqueeze(1)
        pos_encoding[0:d_model:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
        pos_encoding[1:d_model:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)

        # 列编码
        pos_h = torch.arange(0, height).unsqueeze(1)
        pos_encoding[d_model::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
        pos_encoding[d_model+1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        return x + self.pos_encoding


def create_sine_position_encoding(embed_dim, num_positions):
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)  # shape: (num_positions, 1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))  # shape: (embed_dim/2,)
    pe = torch.zeros(num_positions, embed_dim)  # shape: (num_positions, embed_dim)

    pe[:, 0::2] = torch.sin(position * div_term)  # even indices (0, 2, 4, ...) for sine
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices (1, 3, 5, ...) for cosine
    return pe


class SinoCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_positions=180, alpha_init=0.01):
        super(SinoCrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        # self.position_encoding = PositionalEncoding2D(embed_dim, 128, 180)
        self.inch = nn.Linear(1, embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.activation = nn.PReLU(num_parameters=embed_dim)
        # self.positional_encoding = nn.Parameter(torch.arange(0, num_positions)/num_positions)
        self.positional_encoding = create_sine_position_encoding(embed_dim, num_positions)

    def forward(self, x_i, i):
        # 添加位置编码
        x_i = self.inch(x_i)
        qkv = self.qkv(x_i)
        query, key, value = qkv.chunk(3, dim=-1)
        # position_encoding = torch.arange(0, 180)/180
        # position_encoding = position_encoding.unsqueeze(0).repeat(128, 1).flatten()[None, :, None].repeat(x.size(0), 1, x.size(-1)).to(x.device).float()
        positional_e = self.positional_encoding[i, :].unsqueeze(0).repeat(x_i.size(0), x_i.size(1), 1).to(x_i.device)
        # positional_e = self.activation(positional_e)
        positional_e = positional_e * torch.sigmoid(self.alpha * positional_e)
        query = query + positional_e
        key = key + positional_e

        # reshape query, key, and value to [seq_len, batch_size, embed_dim] for multihead attention
        query = query.flatten(2).permute(1, 0, 2)  # shape [seq_len, batch_size, embed_dim]
        key = key.flatten(2).permute(1, 0, 2)
        value = value.flatten(2).permute(1, 0, 2)

        # 计算交叉注意力
        attn_output, _ = self.attention(query, key, value)
        attn_output = self.out(attn_output)
        return attn_output.permute(1, 0, 2) # reshape to [batch_size, 128, 180]


if __name__ == '__main__':
    import torch
    from timm.models.swin_transformer import SwinTransformer

    # 配置参数
    img_size = (128, 180)  # 输入图像尺寸
    patch_size = (1, 45)  # Patch 尺寸
    model = SwinTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1,  # 单通道输入
        embed_dim=96,  # 嵌入维度
        depths=[2, 4, 2],  # 深度配置
        num_heads=[3, 6, 12],  # 注意力头数量
        window_size=(1, 4),  # 窗口大小
        num_classes=0
    )

    # 输入测试
    x = torch.randn(1, 1, 128, 180)  # (Batch, Channel, Height, Width)
    y = model(x)

    # 输出每阶段特征图分辨率
    for stage, feature in enumerate(model.forward_features(x)):
        print(f"Stage {stage} feature size: {feature.shape}")

    # 测试代码
    batch_size = 1
    ch = 64
    num_heads = 8
    # 随机生成 (128, 180) 大小的 query, key, value 特征图
    # x = torch.rand(batch_size, 1, 128, 180)
    # x = x.flatten(2).permute(0, 2, 1)  # 转换为 [batch_size, seq_len, embed_dim=1]
    file_data = np.load('/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_sinoHD.npy', allow_pickle=True)[:4, :, :]
    x = torch.from_numpy(file_data).float().unsqueeze(-1)
    # x_single_angular = [x[:, :, i, :] for i in range(x.size(-2))]
    # x_single_radical = [x[:, i, :, :] for i in range(x.size(-3))]
    # 初始化交叉注意力模块
    # 设置直方图的参数
    bins = 15  # 设置直方图的区间数
    # hist_range = (0, 1)  # 假设数据在 [0, 1] 范围内

    # 计算每个样本的直方图
    for i in range(4):
        raw = x[i]
        hist_range = (x[i].flatten().min(), x[i].flatten().max())
        hist = torch.histc(raw, bins=bins, min=hist_range[0], max=hist_range[1])
        plt.plot(hist.numpy(), label=f'Sample {i + 1}')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    cross_attention_angular = SinoCrossAttention(ch, num_heads, num_positions=180)
    cross_attention_radical = SinoCrossAttention(ch, num_heads, num_positions=128)
    # 计算交叉注意力
    # for i, x in enumerate(x_single_angular):
    #     x_angular = cross_attention_angular(x, i) if i == 0 else torch.cat((x_angular, cross_attention_angular(x, i)), dim=-1)
    # for i, x in enumerate(x_single_radical):
    #     x_radical = cross_attention_radical(x, i) if i == 0 else torch.cat((x_radical, cross_attention_radical(x, i)), dim=-1)

    x_angular_list = [cross_attention_angular(x[:, :, i, :], i) for i in range(x.size(-2))]
    x_angular = torch.cat(x_angular_list, dim=-1)
    x_radical_list = [cross_attention_radical(x[:, i, :, :], i) for i in range(x.size(-3))]
    x_radical = torch.cat(x_radical_list, dim=-1).transpose(-1, -2)

    print("输出大小：", x_angular.shape)  # 输出大小：[batch_size, 128, 180]
