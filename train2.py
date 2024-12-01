import logging
import time
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DistributedSampler, DataLoader, random_split
import matplotlib.pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils.radon import Radon
from utils.data import DatasetPETRecon, tv_loss
from utils.data import load_data, generate_mask
# from recon_astraFBP import sino2pic as s2p
from model.whole_network import PETReconNet, PETDenoiseNet
# from utils.transfer_si import s2i, i2s, s2i_batch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from model.whole_network import normalization2one
# from model.network_swinTrans import SwinIR
from modelSwinUnet.SUNet import SUNet_model


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def simulate_geometry(device):
    temPath = './tmp_1'
    PET = BuildGeometry_v4('mmr', device, 0.5)  # scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    return PET


def train(model_pre, model_recon, radon, train_loader, criterion, optimizer, rank):
    model_pre.train()
    model_recon.train()
    running_loss = 0.0

    for iteration, (inputs, Y, sino_label) in enumerate(train_loader):

        # if (iteration % 10 == 0) and sign:
        time_s = time.time()
        #     sign = not sign
        x1, x2 = inputs
        x1, x2 = x1.to(rank), x2.to(rank)

        # print(torch.cuda.memory_summary())

        # sinogram去噪，noise2noise训练
        x1_denoised = model_pre(x1)
        # x2_denoised = model_pre(x2)
        x2_denoised = x2
        # 平均输出的sinogram
        aver_x = (x1_denoised + normalization2one(x2_denoised)) / 2

        # PET图去噪

        mask_p1, mask_p2 = generate_mask(aver_x.shape, 0.4)
        mask_p1, mask_p2 = torch.from_numpy(mask_p1).unsqueeze(1).float().to(rank), torch.from_numpy(mask_p2).unsqueeze(
            1).float().to(rank)
        sino_m1, sino_m2 = aver_x * mask_p1, aver_x * mask_p2
        pic_in_m1, pic_in_m2 = radon.filter_backprojection(sino_m1), radon.filter_backprojection(sino_m2)
        pic_recon_m1, pic_recon_m2 = model_recon(normalization2one(pic_in_m1), aver_x, mask_p1), model_recon(normalization2one(pic_in_m2), aver_x, mask_p2)
        aver_recon_pic = (pic_recon_m1 + pic_recon_m2)/2


        # 计算mask角度下的loss
        sino_recon_m1, sino_recon_m2, aver_recon_sino = radon(pic_recon_m1), radon(pic_recon_m2), radon(aver_recon_pic)
        # sino_recon_m1, sino_recon_m2 = (sino_recon_m1 + aver_x) / 2, (sino_recon_m2 + aver_x) / 2
        # aver_recon_sino = (aver_recon_sino + aver_x) / 2  # 加个残差, aver_x是输入
        # sino_recon_p1 = sino_recon_p1[None, None, :, :] if len(sino_recon_p1.shape) == 2 else sino_recon_p1[:, None, :, :]
        # sino_recon_p2 = sino_recon_p2[None, None, :, :] if len(sino_recon_p2.shape) == 2 else sino_recon_p2[:, None, :, :]
        sino_recon_m1_m2, sino_recon_m2_m1 = sino_recon_m1 * mask_p2, sino_recon_m2 * mask_p1
        # sino_recon_m1_m2, sino_recon_m2_m1 = normalization2one(sino_recon_m1_m2), normalization2one(sino_recon_m2_m1)
        sino_recon_m1_m2, sino_recon_m2_m1 = normalization2one(sino_recon_m1_m2), normalization2one(sino_recon_m2_m1)

        lsp1, lsp2 = criterion(sino_recon_m1_m2, sino_m2), criterion(sino_recon_m2_m1, sino_m1)
        lspre = criterion(x1_denoised, x2_denoised) + 0.1*tv_loss(x1_denoised) + 0.1*tv_loss(x2_denoised)
        li = criterion(pic_recon_m1, pic_recon_m2)
        loss = lspre + lsp1 + lsp2 + li

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cur = loss.item()
        running_loss += loss_cur
        if iteration % 100 == 0:
            logger.info(f'Iteration: {iteration}/{len(train_loader)}, Loss: {loss_cur:.4f}, Time/p_i: {time.time() - time_s:.4f}')
            # for i, pic in enumerate((x1, x2, x1_denoised, x2_denoised, aver_x, aver_x, mask_p1, mask_p2, sino_p1, sino_p2, pic_recon_p1, pic_recon_p2)):
            #     plt.subplot(6, 2, i + 1), plt.title(i)
            #     plt.imshow(pic[0, 0, :, :].cpu().detach().numpy()), plt.title(f'{["x1", "x2", "x1_denoised", "x2_denoised", "aver_x", "aver_x", "mask_p1", "mask_p2", "sino_p1", "sino_p2", "pic_recon_p1", "pic_recon_p2"][i]}')
            # plt.suptitle(f'Iteration: {iteration}, lspre: {lspre:.4f}, lsp1: {lsp1:.4f}, lsp2: {lsp2:.4f}, li: {li:.4f}'), plt.show()
            # #
            # 定义图像数据和标题
            pics = [x1, x2, x1_denoised, x2_denoised, aver_x, aver_x, sino_m1, sino_m2, sino_recon_m2_m1,
                    sino_recon_m1_m2, pic_recon_m1, pic_recon_m2, aver_recon_sino, sino_label, aver_recon_pic, Y]
            titles = ["x1", "x2", "x1_denoised", "x2_denoised", "aver_x", "aver_x", "sino_p1",
                      "sino_p2", "sino_recon_p1", "sino_recon_p2", 'pic_recon_m1', 'pic_recon_m2', 'aver_recon_sino', 'label_sino', 'aver_recon_pic', 'label']
            # 设置图像整体大小
            fig, axes = plt.subplots(8, 2, figsize=(12, 18))
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
            fig.suptitle(
                f'Iteration: {iteration}, lspre: {lspre:.4f}, lsp1: {lsp1:.4f}, lsp2: {lsp2:.4f}, li: {li:.4f}',
                fontsize=14)
            # 调整布局，避免重叠

            plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'/home/ssddata/linshuijin/PETrecon/log_file/pic_visual_1/{iteration}_{loss_cur:.4f}.png', format='png', dpi=300, bbox_inches='tight')  # 300 dpi保证高分辨率

            # 显示图像
            plt.show()
        # logger.info(f'Train Loss: {running_loss:.4f}')

    loss_average = running_loss / len(train_loader)
    return loss_average


def validate(model_pre, model_recon, radon, val_loader, criterion, rank):
    model_pre.eval()
    model_recon.eval()
    running_loss = 0.0

    with torch.no_grad():
        for iteration, (inputs, Y, _) in enumerate(val_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float()

            # sinogram去噪，noise2noise训练
            x1_denoised = model_pre(x1)
            x2_denoised = model_pre(x2)
            # x2_denoised = x2
            # 平均输出的sinogram
            aver_x = (x1_denoised + x2_denoised) / 2

            # PET图去噪

            mask_p1, mask_p2 = generate_mask(aver_x.shape, 0.01)
            mask_p1, mask_p2 = torch.from_numpy(mask_p1).unsqueeze(1).float().to(rank), torch.from_numpy(mask_p2).unsqueeze(
                1).float().to(rank)
            sino_p1, sino_p2 = aver_x * mask_p2, aver_x * mask_p1
            pic_in_p1, pic_in_p2 = radon.filter_backprojection(sino_p1), radon.filter_backprojection(sino_p2)
            pic_recon_p1, pic_recon_p2 = model_recon(pic_in_p1, aver_x, mask_p1), model_recon(pic_in_p2, aver_x,
                                                                                              mask_p2)
            pic_recon = (pic_recon_p1 + pic_recon_p2)/2
            loss = criterion(pic_recon, Y)
            running_loss += loss.item()
            if iteration % 10 == 0:
                logger.info(f'Iteration: {iteration}/{len(val_loader)}, Validation Loss: {loss.item():.4f}')
        avg_loss = running_loss / len(val_loader)
        return avg_loss


def test(model_pre, model_recon, radon, test_loader, criterion, rank):
    model_pre.eval()
    model_recon.eval()
    running_loss = 0.0

    with torch.no_grad():
        for iteration, (inputs, Y, _) in enumerate(test_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float()

            # sinogram去噪，noise2noise训练
            x1_denoised = model_pre(x1)
            x2_denoised = model_pre(x2)
            # x2_denoised = x2
            # 平均输出的sinogram
            aver_x = (x1_denoised + x2_denoised) / 2

            # PET图去噪

            mask_p1, mask_p2 = generate_mask(aver_x.shape, 0.01)
            mask_p1, mask_p2 = torch.from_numpy(mask_p1).unsqueeze(1).float().to(rank), torch.from_numpy(mask_p2).unsqueeze(
                1).float().to(rank)
            sino_p1, sino_p2 = aver_x * mask_p2, aver_x * mask_p1
            pic_in_p1, pic_in_p2 = radon.filter_backprojection(sino_p1), radon.filter_backprojection(sino_p2)
            pic_recon_p1, pic_recon_p2 = model_recon(pic_in_p1, aver_x, mask_p1), model_recon(pic_in_p2, aver_x,
                                                                                              mask_p2)
            pic_recon = (pic_recon_p1 + pic_recon_p2) / 2
            loss = criterion(pic_recon, Y)
            running_loss += loss.item()
            if iteration % 10 == 0:
                logger.info(f'Iteration: {iteration}/{len(test_loader)}, Test Loss: {loss.item():.4f}')
        avg_loss = running_loss / len(test_loader)
        return avg_loss


def main(logger, file_path, n_theta, config):
    # # 数据
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radon = Radon(n_theta, circle=True, device=rank)

    dataset = DatasetPETRecon(file_path, radon, ratio=0.2)

    # 将数据集按80/10/10比例划分为训练集、验证集和测试集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 使用DataLoader加载数据
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 模型初始化

    denoise_model_pre = PETDenoiseNet(device=rank)
    denoise_model = PETReconNet(radon, device=rank, config=config)

    # print(torch.cuda.memory_summary())

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(denoise_model.parameters()) + list(denoise_model_pre.parameters()), lr=2e-4)

    # 训练
    num_epochs = 50
    signal = 0.2
    for epoch in range(num_epochs):
        logger.info('start train !')
        train_loss = train(denoise_model_pre, denoise_model, radon, train_loader, criterion, optimizer, rank)
        logger.info('start validate !')
        val_loss = validate(denoise_model_pre, denoise_model, radon, val_loader, criterion, rank)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < signal:
            signal = val_loss
            torch.save(denoise_model_pre.state_dict(), f'./model/denoise_pre_weight.pth')
            torch.save(denoise_model.state_dict(), f'./model/denoise_weight.pth')
    logger.info('start test !')
    test_loss = test(denoise_model_pre, denoise_model, radon, test_loader, criterion, rank)
    logger.info(f'Test Loss: {test_loss:.4f}')



if __name__ == '__main__':
    n_theta = 180
    recon_size = 128
    # 配置logging模块
    log_file_path = './log_file/training_log_1107plus.txt'

    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    # 创建一个handler，用于将日志写入文件
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # 再创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将两个handler都添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # temPath = f'./tmp_{n_theta}_{recon_size}*{recon_size}/'
    # PET = BuildGeometry_v4('mmr', 0)  # scanner mmr, with radial crop factor of 50%
    # PET.loadSystemMatrix(temPath, is3d=False)
    #
    # geoMatrix = []
    # geoMatrix.append(np.load(temPath + 'geoMatrix-0.npy', allow_pickle=True))

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 绕过 GPU 0，只使用 GPU 1 和 GPU 2
    # world_size = torch.cuda.device_count()
    path = f'./simulation_angular/angular_{n_theta}'
    # os.environ['MASTER_ADDR'] = '10.181.8.117'
    # os.environ['MASTER_PORT'] = '12345'
    with open('./modelSwinUnet/training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    main(logger, path, n_theta, opt)
