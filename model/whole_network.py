import numpy as np
import torch
import timm
from timm.models.swin_transformer import SwinTransformer
from torch import nn
import torch.nn.functional as F

# from geometry.BuildGeometry_v4 import BuildGeometry_v4
from model.network_swinTrans import SwinIR
# from recon_astraFBP import sino2pic as s2p
# from utils.transfer_si import i2s, s2i, s2i_batch
from modelSwinUnet.SUNet import SUNet_model
from model.model_sino import SinoCrossAttention

import torch
import torch.nn.functional as F


class SwinDenoise(nn.Module):
    def __init__(self, img_size=(128, 180), in_chans=1, embed_dim=96, depths=[2, 6], num_heads=[3, 6]):
        """
        Swin Transformer-based Denoising Model with custom patch size (1×180).
        Parameters:
        - img_size: 输入图像大小 (H, W)
        - in_chans: 输入图像的通道数
        - embed_dim: 嵌入维度
        - depths: 每个阶段的 Transformer Block 数量
        - num_heads: 每个阶段的注意力头数量
        """
        super(SwinDenoise, self).__init__()

        self.img_size = img_size
        self.patch_size = (1, 45)  # 设置为 (1×180)

        # Swin Transformer Backbone
        self.backbone = SwinTransformer(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            num_classes=0,  # 设置为 0，表示没有分类头
            window_size=(1, 4)
        )

        # 输出层：恢复到原始图像大小
        self.output_layer = nn.Conv2d(
            in_channels=embed_dim,  # 嵌入维度
            out_channels=in_chans,  # 恢复为输入通道数
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        """
        Forward pass of the Swin Transformer-based denoising network.
        x: 输入图像, 形状为 [B, C, H, W]
        输出: 去噪后的图像, 形状为 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Backbone processing
        x = self.backbone.patch_embed(x)  # 提取 patch 嵌入
        x = self.backbone.layers(x)  # Swin Transformer 层处理

        # 恢复到二维特征图
        x = x.transpose(1, 2).reshape(B, -1, H // self.patch_size[0], W // self.patch_size[1])

        # 输出去噪结果
        x = self.output_layer(x)
        return x


class CorrectMLP(nn.Module):
    def __init__(self, mean):
        super(CorrectMLP, self).__init__()
        self.mean = torch.tensor(mean)

    def forward(self, x: torch.Tensor):
        assert len(x.size()) == 4
        y = normalization2one(x)
        return y - self.mean.to(x.device)


class BaseMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BaseMLP, self).__init__()

        # 输入层到第一个隐藏层
        correctNet = CorrectMLP(mean=0.5)
        layers = [nn.Linear(input_size, hidden_sizes[0]), correctNet, nn.Sigmoid()]

        # 构建隐藏层
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(correctNet)  # correctNet将数据归一化并减去均值，数据取值范围在-0.5到0.5之间
            layers.append(nn.Sigmoid())

        # 添加输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # 将 layers 列表传入 nn.Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RowAttention(torch.nn.Module):
    def __init__(self, input_dim, size):
        super(RowAttention, self).__init__()
        self.input_dim = input_dim
        self.qkv = torch.nn.Linear(input_dim, input_dim * 3, bias=False)
        self.r, self.a = size

    def forward(self, x):
        """
        x: 输入数据，形状为 (bs, h*w, 1)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, self.r, self.a, 3).permute(3, 0, 1, 2)

        # Step 1: 计算查询、键、值
        query, key, value = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Step 2: 计算行级别的注意力
        # 对每一行，计算 Query 和 Key 的点积
        query = query.permute(0, 2, 1)  # (bs, w, h)
        attention_scores = torch.bmm(query, key)  # (bs, w, w)

        # Step 3: 使用 Softmax 正规化注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (bs, w, w)

        # Step 4: 使用注意力权重加权值
        output = torch.bmm(attention_weights, value.permute(0, 2, 1))  # (bs, w, h)
        output = output.permute(0, 2, 1)  # (bs, h, w)

        return output


class ColumnAttention(torch.nn.Module):
    def __init__(self, input_dim, size):
        super(ColumnAttention, self).__init__()
        self.input_dim = input_dim
        self.qkv = torch.nn.Linear(input_dim, input_dim * 3, bias=False)
        self.r, self.a = size

    def forward(self, x):
        """
        x: 输入数据，形状为 (bs, h, w)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, self.r, self.a, 3).permute(3, 0, 1, 2)

        # Step 1: 计算查询、键、值
        query, key, value = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Step 2: 计算列级别的注意力
        # 对每一列，计算 Query 和 Key 的点积
        query = query.permute(0, 1, 2)  # (bs, h, w)
        attention_scores = torch.bmm(query, key.permute(0, 2, 1))  # (bs, h, h)

        # Step 3: 使用 Softmax 正规化注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (bs, h, h)

        # Step 4: 使用注意力权重加权值
        output = torch.bmm(attention_weights, value)  # (bs, h, w)

        return output


def otsu_threshold_batch(img_batch):
    # Step 1: 计算每个图像的直方图（在灰度值范围0-255）
    batch_size, channels, height, width = img_batch.shape
    hist = torch.stack([torch.histc(img_batch[i], bins=256, min=0, max=255) for i in range(batch_size)],
                       dim=0)  # shape: (batch_size, 256)
    hist = hist / hist.sum(dim=1, keepdim=True)  # 归一化直方图，每个图像的灰度分布

    # Step 2: 计算累积和和累积均值
    cumsum_hist = torch.cumsum(hist, dim=1)  # 累积和，shape: (batch_size, 256)
    cumsum_mean = torch.cumsum(hist * torch.arange(256, device=img_batch.device),
                               dim=1)  # 累积均值，shape: (batch_size, 256)
    global_mean = cumsum_mean[:, -1]  # 全局均值，shape: (batch_size,)

    # Step 3: 计算类间方差
    numerator = (global_mean.unsqueeze(1) * cumsum_hist - cumsum_mean) ** 2
    denominator = cumsum_hist * (1 - cumsum_hist)
    between_class_variance = numerator / (denominator + 1e-6)  # 避免除零

    # Step 4: 获取最大方差对应的阈值
    _, optimal_thresholds = torch.max(between_class_variance, dim=1)  # shape: (batch_size,)

    # Step 5: 根据最优阈值生成掩膜
    optimal_thresholds = optimal_thresholds.view(batch_size, 1, 1, 1).expand(-1, channels, height, width)  # 调整阈值形状
    mask_batch = (img_batch >= optimal_thresholds).float()  # 将掩膜转换为0和1的浮点型结果

    return mask_batch


def normalization2one(input_tensor):
    # 假设输入的 tensor 形状为 (batchsize, channels=1, h, w)
    # input_tensor = torch.randn(4, 1, 64, 64)  # 示例的输入张量

    # 为了每个 batch 归一化，我们要按batch维度进行最小值和最大值的计算
    # 计算每个batch的最小值和最大值，保持维度为 (batchsize, 1, 1, 1)
    min_val = input_tensor.reshape(input_tensor.size(0), -1).min(dim=1)[0].reshape(-1, 1, 1, 1)
    max_val = input_tensor.reshape(input_tensor.size(0), -1).max(dim=1)[0].reshape(-1, 1, 1, 1)

    # 进行归一化，将所有数值归一化到 [0, 1] 区间
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + 1e-8)  # 1e-8 防止除以0
    assert input_tensor.shape == normalized_tensor.shape

    return normalized_tensor  # 确认输出形状 (batchsize, 1, h, w)


class PETReconNet(nn.Module):
    def __init__(self, radon, device, config, num_block=3):
        super().__init__()
        self.num_block = num_block
        self.radon = radon
        # self.n_theta = 2 * self.geo[0].shape[0]
        self.device = device
        # self.PET = PET
        # self.norm1 = nn.BatchNorm2d(1)
        # self.norm2 = nn.BatchNorm2d(1)
        # self.norm3 = nn.BatchNorm2d(1)
        # self.denoiseBlock1 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)
        self.denoiseBlock1 = SUNet_model(config).to(self.device)
        self.denoiseBlock2 = SUNet_model(config).to(self.device)
        # self.denoiseBlock3 = SUNet_model(config).to(self.device)
        # self.denoiseBlock2 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)
        # self.denoiseBlock3 = SwinIR(img_size, depths=[3, 3], num_heads=[4, 4]).to(device)

    def forward(self, image_p, sino_o, mask):
        image = normalization2one(image_p)
        image = self.denoiseBlock1(image)
        image = normalization2one(image)
        image = self.DCLayer(image, mask, sino_o)
        image = normalization2one(image)
        # image = self.denoiseBlock2(image)
        # image = normalization2one(image)

        return image

    def DCLayer(self, x_p, mask, sino_o):
        sino_re = self.radon.forward(x_p, )
        sino_re = sino_re.to(self.device)
        # sino_re = sino_re[:, None, :, :]
        out_sino = normalization2one(sino_o) * (1 - mask) + normalization2one(sino_re) * mask

        out_pic = self.radon.filter_backprojection(out_sino)
        # if out_sino.shape[0] == 1:
        #     out_sino = s2i(out_sino)
        #     # out_sino = out_sino[None, None, :, :]
        #     out_sino = out_sino.unsqueeze([0, 1])
        #     return out_sino
        # else:
        #     sino_t = []
        #     for sino in out_sino:
        #         sino_t.append(s2i(sino))
        #     out_sino = torch.stack(sino_t, 0)
        #     out_sino = out_sino.unsqueeze([0, 1])
        return out_pic


class PETDenoiseNet(nn.Module):
    def __init__(self, device, num_block=3, raw_size=(128, 180)):
        super().__init__()
        self.num_block = num_block
        self.correctNet = CorrectMLP(mean=0.5)
        self.mean = torch.tensor(0.5).to(device)
        a_delta, r_delta = torch.tensor(0.3).to(device), torch.tensor(0.3).to(device)
        # a_delta, r_delta = nn.Parameter(torch.tensor(0.3)).to(device), nn.Parameter(torch.tensor(0.3)).to(device)
        self.angularEncode = a_delta * normalization2one(
            torch.deg2rad(torch.arange(0, 180).float())[None, None, None, :]).to(device)
        self.intoAngular = nn.Linear(180, 180).to(device)
        self.crossAngular = BaseMLP(input_size=180, hidden_sizes=[128, 64, 128], output_size=180).to(device)
        # self.crossAngular = RowAttention(input_dim=1, size=raw_size).to(device)
        # self.crossRadical = ColumnAttention(input_dim=1, size=raw_size).to(device)
        self.radicalEncode = r_delta * normalization2one((torch.arange(0, 128).float())[None, None, None, :]).to(device)
        self.intoRadical = nn.Linear(128, 128).to(device)
        self.crossRadical = BaseMLP(input_size=128, hidden_sizes=[64], output_size=128).to(device)

        self.concatOut = nn.Conv2d(2, 1, 1).to(device)
        # self.denoiseBlock3 = SwinIR(img_size=168)
        # 添加 BatchNorm 层
        self.in_batchNorm = nn.BatchNorm2d(num_features=1).to(device)
        self.out_batchNorm = nn.BatchNorm2d(num_features=1).to(device)
        self.activation = nn.ReLU().to(device)
        # self.radical_norm = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        # x是弦图, shape: (batch_size, channel, 128, 180)
        # bs, c, h, w = x.size()
        x = normalization2one(x)
        raw = x.clone().transpose(-1, -2)
        # t, m = self.intoAngular(x), self.intoAngular(self.angularEncode)
        # t_ = [t.cpu().detach().flatten().numpy().min(), t.cpu().detach().flatten().numpy().max()]
        # m_ = [m.cpu().detach().flatten().numpy().min(), m.cpu().detach().flatten().numpy().max()]
        embed_angular_x = normalization2one(self.intoAngular(x)) + normalization2one(
            self.intoAngular(self.angularEncode))
        x_angular = self.crossAngular(embed_angular_x)
        embed_radical_x = normalization2one(self.intoRadical(raw)) + normalization2one(
            self.intoRadical(self.radicalEncode))
        x_radical = self.crossRadical(embed_radical_x).transpose(-1, -2)

        # x_out = x_angular
        x_out = self.concatOut(torch.cat((x_angular, x_radical), dim=1))
        # x_out = self.out_batchNorm(x_out)
        # x_out = self.activation(x_out)
        # mask_tem = otsu_threshold_batch(255*image_p)
        # image = self.denoiseBlock1(image_p)
        sino_out = normalization2one(x_out)
        # image = mask_tem * image
        return sino_out
