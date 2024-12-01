import time

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DistributedSampler, DataLoader
import matplotlib.pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils import Radon
from utils.data import DatasetPETRecon, tv_loss
from utils.data import load_data, generate_mask
from recon_astraFBP import sino2pic as s2p
from model.whole_network import PETReconNet, PETDenoiseNet
# from utils.transfer_si import s2i, i2s, s2i_batch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from model.whole_network import normalization2one
# from model.network_swinTrans import SwinIR
from modelSwinUnet.SUNet import SUNet_model
import logging


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def simulate_geometry(device):
    temPath = './tmp_1'
    PET = BuildGeometry_v4('mmr', device, 0.5)  # scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    return PET


def main(logger, file_path, n_theta, config):
    # # 数据
    # setup(rank, world_size)
    # dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    train_set = DatasetPETRecon(file_path, 'train')
    # train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True, seed=seed)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

    # 模型初始化
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radon = Radon(n_theta, circle=True, device=rank)

    # PET = simulate_geometry(device)

    # denoise_model_pre = SUNet_model(config).to(rank)
    denoise_model_pre = PETDenoiseNet(device=rank)
    denoise_model = PETReconNet(radon, device=rank, config=config)

    # print(torch.cuda.memory_summary())

    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(denoise_model.parameters()) + list(denoise_model_pre.parameters()), lr=1e-4)

    # 训练
    num_epochs = 100
    for epoch in range(num_epochs):
        denoise_model.train()
        denoise_model_pre.train()
        running_loss = 0.0
        for iteration, (inputs, Y) in enumerate(train_loader):
            # if (iteration % 10 == 0) and sign:
            time_s = time.time()
            #     sign = not sign
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)

            # print(torch.cuda.memory_summary())

            # sinogram去噪，noise2noise训练
            x1_denoised = denoise_model_pre(x1)
            x2_denoised = denoise_model_pre(x2)
            # 平均输出的sinogram
            aver_x = (x1_denoised + x2_denoised) / 2.

            # PET图去噪

            mask_p1, mask_p2 = generate_mask(aver_x.shape, 0.01)
            mask_p1, mask_p2 = torch.from_numpy(mask_p1).unsqueeze(1).float().to(rank), torch.from_numpy(mask_p2).unsqueeze(1).float().to(rank)
            sino_p1, sino_p2 = aver_x * mask_p2, aver_x * mask_p1
            pic_in_p1, pic_in_p2 = radon.filter_backprojection(sino_p1), radon.filter_backprojection(sino_p2)
            pic_recon_p1, pic_recon_p2 = denoise_model(pic_in_p1, aver_x, AN, mask_p1), denoise_model(pic_in_p2, aver_x, AN, mask_p2)

            # 计算mask角度下的loss
            sino_recon_p1, sino_recon_p2 = radon.forward(pic_recon_p1), radon.forward(pic_recon_p2)
            sino_recon_p1 = sino_recon_p1[None, None, :, :] if len(sino_recon_p1.shape) == 2 else sino_recon_p1[:, None, :, :]
            sino_recon_p2 = sino_recon_p2[None, None, :, :] if len(sino_recon_p2.shape) == 2 else sino_recon_p2[:, None, :, :]
            sino_recon_p1_m2, sino_recon_p2_m1 = sino_recon_p1 * mask_p2, sino_recon_p2 * mask_p1
            sino_recon_p1_m2, sino_recon_p2_m1 = normalization2one(sino_recon_p1_m2), normalization2one(
                sino_recon_p2_m1)

            lsp1, lsp2 = criterion(sino_recon_p1_m2, sino_p2), criterion(sino_recon_p2_m1, sino_p1)
            lspre = criterion(x1_denoised, x2_denoised) + tv_loss(x1_denoised) + tv_loss(x2_denoised)
            li = criterion(pic_recon_p1, pic_recon_p2)
            loss = lspre + lsp1 + lsp2 + li

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Iteration {iteration}/{int(np.ceil(len(train_loader)/x1.size(0)))}, Loss: {running_loss:.4f}, Time: {time.time()-time_s:.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    n_theta = 180
    recon_size = 128
    # 配置logging模块
    log_file_path = './log_file/training_log.txt'

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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 绕过 GPU 0，只使用 GPU 1 和 GPU 2
    # world_size = torch.cuda.device_count()
    path = f'./simulation_angular/angular_{n_theta}'
    # os.environ['MASTER_ADDR'] = '10.181.8.117'
    # os.environ['MASTER_PORT'] = '12345'
    with open('/mnt/data/linshuijin/PETrecon/modelSwinUnet/training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    main(logger, path, n_theta, opt)
