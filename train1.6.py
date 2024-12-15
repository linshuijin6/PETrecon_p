import argparse
import logging
import random
import time
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn, optim
from torch.utils.data import DistributedSampler, DataLoader, random_split
import matplotlib.pyplot as plt

from filter_design.wavelet_trans import WaveletFilterNet
from geometry.BuildGeometry_v4 import BuildGeometry_v4
# from utils.radon import Radon
from torch_radon import Radon
from utils.data import DatasetPETRecon, tv_loss
from utils.data import load_data, generate_mask
# from recon_astraFBP import sino2pic as s2p
from model.whole_network import PETReconNet, PETDenoiseNet, SwinDenoise
# from utils.transfer_si import s2i, i2s, s2i_batch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from utils.data import normalization2one
# from model.network_swinTrans import SwinIR
from modelSwinUnet.SUNet import SUNet_model
from torch.utils.tensorboard import SummaryWriter
from result_eval.evaluate import calculate_metrics


def preset_seed(seed_number):
    import torch.backends.cudnn
    seed = seed_number
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_gen(args):
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    args.log_dir = f'./log/log_file_{seed}'
    log_file_path = f'./log/log_file_{seed}/log_file_{seed}.txt'
    os.mkdir(args.log_dir) if not os.path.exists(args.log_dir) else None

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
    logger.info(f'seed = {seed}')
    logger.info(args)
    return logger


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def simulate_geometry(device):
    temPath = './tmp_1'
    PET = BuildGeometry_v4('mmr', device, 0.5)  # scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    return PET


def train(model_pre, model_recon, radon, train_loader, criterion, optimizer, rank, epoch, logger, log_writer):
    model_pre.train()
    model_recon.train()
    running_loss = 0.0

    for iteration, (inputs, Y, sino_label, picLD) in enumerate(train_loader):

        # if (iteration % 10 == 0) and sign:

        time_s = time.time()
        #     sign = not sign
        x1, x2 = inputs
        x1, x2 = x1.to(rank), x2.to(rank)
        bs = x1.shape[0]
        # print(torch.cuda.memory_summary())

        # sinogram去噪，noise2noise训练
        x1_denoised = model_pre(x1)
        # x2_denoised = model_pre(x2)

        x2_denoised = normalization2one(x2)
        # 平均输出的sinogram
        # aver_x2 = (x1_denoised + x2_denoised) / 2
        aver_x2 = x1_denoised
        w = torch.sigmoid(10 * (x1_denoised - torch.tensor(0.5, device=x1_denoised.device))).detach()  #将灰度值从0~1平移到-0.5~0.5，使得sigmoid更恰当
        aver_x1 = (normalization2one(x1_denoised * w) + x2_denoised) / 2
        aver_x = model_recon(aver_x1)
        mid_recon1 = normalization2one(radon.backprojection(radon.filter_sinogram(aver_x1)))
        mid_recon2 = normalization2one(radon.backprojection(radon.filter_sinogram(aver_x2)))
        mid_recon = normalization2one(radon.backprojection(radon.filter_sinogram(aver_x)))

        # 计算mask角度下的loss

        mask_p1, mask_p2 = generate_mask(mid_recon.shape, 0.5)

        mask_p1, mask_p2 = torch.from_numpy(mask_p1).unsqueeze(1).float().to(rank), torch.from_numpy(mask_p2).unsqueeze(
            1).float().to(rank)
        i_in_m1, i_in_m2 = mid_recon * mask_p1, mid_recon * mask_p2
        s_in_1, s_in_2 = radon.forward(i_in_m1), radon.forward(i_in_m2)
        s_out_1, s_out_2 = model_pre(s_in_1), model_pre(s_in_2)
        i_out_1, i_out_2 = radon.backprojection(radon.filter_sinogram(s_out_1)), radon.backprojection(
            radon.filter_sinogram(s_out_2))
        i_out_m12, i_out_m21 = i_out_1 * mask_p2, i_out_2 * mask_p1
        lsm1, lsm2 = criterion(normalization2one(i_out_m12), normalization2one(i_in_m2)), criterion(
            normalization2one(i_out_m21), normalization2one(i_in_m1))
        lsi = criterion(normalization2one(i_out_1), normalization2one(i_out_2))
        lsm = lsm1 + lsm2
        ls_post = lsm + lsi
        ls_sino = criterion(x1_denoised, x2_denoised)
        loss = ls_sino + ls_post + args.alpha * tv_loss(x1_denoised) - args.beta * tv_loss(mid_recon)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in model_pre.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).sum() > 0:
                print(f"NaN detected in gradient of {name}")

        loss_cur = loss.item() / bs
        lossm_cur = lsm.item() / bs
        lossi_cur = lsi.item() / bs
        running_loss += loss_cur
        if iteration % 100 == 0:
            logger.info(
                f'Epoch:{epoch}, Iteration: {iteration}/{len(train_loader)}, a_loss: {loss_cur:.4f}, loss_m: {lossm_cur:.4f}, loss_i: {lossi_cur:.4f}, Time/p_i: {time.time() - time_s:.4f}')

            # 定义图像数据和标题
            pics = [x1, x2, x1_denoised, aver_x, picLD, mid_recon,
                    aver_x1, aver_x2, mid_recon1, mid_recon2, sino_label, Y]
            titles = ["x1", "x2", "x1_denoised", "aver_x", "input_LD", "mid_recon",
                      "aver_x1", "aver_x2", "mid_recon1", "mid_recon2", "sino_label", "Y"]
            # 设置图像整体大小
            fig, axes = plt.subplots(6, 2, figsize=(12, 18))

            # # 定义图像数据和标题
            # pics = [x1, x2, x1_denoised, aver_x, picLD, mid_recon1, s_in_1, s_in_2, s_out_1, s_out_2,
            #         i_out_m12, i_in_m2, i_out_m21, i_in_m1, aver_x1, sino_label, mid_recon, Y]
            # titles = ["x1", "x2", "x1_denoised", "aver_x", "input_LD", "mid_recon1", "sino_p1", "sino_p2",
            #           "sino_recon_p1", "sino_recon_p2",
            #           'i_out_m12', 'i_in_m2', 'i_out_m21', 'i_in_m1', 'aver_recon_sino', 'sino_label', 'pic_recon',
            #           'pic_label']
            # # 设置图像整体大小
            # fig, axes = plt.subplots(9, 2, figsize=(12, 18))

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
                f'Epoch:{epoch}, Iteration: {iteration}, a_loss: {loss_cur:.4f}, loss_m: {lossm_cur:.4f}, loss_i: {lossi_cur:.4f}, seed: {seed}',
                fontsize=14)
            # 调整布局，避免重叠

            plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
            os.mkdir(args.log_dir) if not os.path.exists(args.log_dir) else None
            os.mkdir(os.path.join(args.log_dir, 'pic_visual')) if not os.path.exists(
                os.path.join(args.log_dir, 'pic_visual')) else None
            pic_path = os.path.join(args.log_dir, 'pic_visual', f'tr_{iteration}_{loss_cur:.4f}_s{seed}.png')
            plt.savefig(pic_path, format='png', dpi=300, bbox_inches='tight')  # 300 dpi保证高分辨率

            if args.show_tr:
                # 显示图像
                plt.show()
                plt.close()
                # 将 matplotlib 图像转换为 PIL 图像
                image = Image.open(pic_path)  # 使用 PIL 打开图像
                image = np.array(image)  # 转换为 NumPy 数组

                # 转换为 Tensor 格式，TensorBoard 需要的形状是 (C, H, W)
                image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # 从 (H, W, C) 转换为 (C, H, W)

                # 记录图像到 TensorBoard
                # log_writer.add_image('train duration', image_tensor, epoch)
        # logger.info(f'Train Loss: {running_loss:.4f}')

    loss_average = running_loss / len(train_loader)
    log_writer.add_scalar('train/loss', loss_average, epoch)  # 在一个 tag 下面添加多个折线图
    return loss_average


def validate(model_pre, model_recon, radon, val_loader, criterion, rank, epoch, logger, log_writer):
    model_pre.eval()
    model_recon.eval()
    running_loss = 0.0

    with torch.no_grad():
        for iteration, (inputs, Y, sino_label, picLD) in enumerate(val_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float()

            # sinogram去噪，noise2noise训练
            x1_denoised = x1
            x2_denoised = model_pre(x2)
            # x2_denoised = x2
            # 平均输出的sinogram
            w = torch.sigmoid(10 * (x2_denoised - torch.tensor(0.5, device=x1_denoised.device))).detach()  # 将灰度值从0~1平移到-0.5~0.5，使得sigmoid更恰当
            aver_xt = normalization2one(x2_denoised * w)

            aver_x = x2_denoised
            aver_x1 = model_recon(aver_x)
            mid_recon = normalization2one(radon.backprojection(radon.filter_sinogram(aver_x1)))
            mid_recont = normalization2one(radon.backprojection(radon.filter_sinogram(aver_xt)))

            # PET图去噪
            # p_out = model_recon(mid_recon, aver_x, torch.ones_like(aver_x))
            # pic_recon = (p_out + mid_recon) / 2
            # sino_recon = radon(pic_recon)

            loss = criterion(mid_recon, Y)
            loss_cur = loss.item() / x1.shape[0]
            running_loss += loss.item()
            if iteration % 40 == 0:
                logger.info(f'Epoch:{epoch}, Validation Loss: {loss_cur:.4f}')
                # 定义图像数据和标题
                pics = [x1, x2, x1_denoised, aver_x, x2_denoised, aver_x, mid_recon, picLD, aver_x, sino_label,
                        mid_recon, Y, aver_xt, mid_recont]
                titles = ["x1", "x2", "x1_denoised", "aver_x", "x2_denoised", "aver_x", "mid_recon", "input_LD",
                          "sino_recon", 'label_sino', 'pic_recon', 'label', 'aver_xt', 'mid_recont']
                # 设置图像整体大小
                fig, axes = plt.subplots(7, 2, figsize=(12, 18))
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
                    f'Validate, Epoch:{epoch}, loss_cur: {loss_cur:.4f}', fontsize=14)
                # 调整布局，避免重叠

                plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
                # os.mkdir(os.path.join(args.log_dir, 'pic_visual')) if not os.path.exists(
                #     os.path.join(args.log_dir, 'pic_visual')) else None
                pic_path = os.path.join(args.log_dir, 'pic_visual', f'val_{iteration}_{loss_cur:.4f}_s{seed}.png')
                # pic_path = f'/home/ssddata/linshuijin/PETrecon/log_file/pic_visual/val_{iteration}_{loss_cur:.4f}.png'
                plt.savefig(pic_path, format='png', dpi=300, bbox_inches='tight')  # 300 dpi保证高分辨率

                if args.show_val:
                    # 显示图像
                    plt.show()
                    plt.close()
                    # 将 matplotlib 图像转换为 PIL 图像
                    image = Image.open(pic_path)  # 使用 PIL 打开图像
                    image = np.array(image)  # 转换为 NumPy 数组

                    # 转换为 Tensor 格式，TensorBoard 需要的形状是 (C, H, W)
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # 从 (H, W, C) 转换为 (C, H, W)

                    # 记录图像到 TensorBoard
                    # log_writer.add_image('val duration', image_tensor, epoch)
        avg_loss = running_loss / len(val_loader)
        log_writer.add_scalar('val/loss', avg_loss, epoch)  # 在一个 tag 下面添加多个折线图
        return avg_loss


def test(model_pre, model_recon, radon, test_loader, criterion, rank, logger, log_writer):
    logger.info('load net parameters...')
    model_pre.load_state_dict(torch.load(os.path.join(args.log_dir, "denoise_pre_weight_best.pth")))
    model_pre.eval()
    model_recon.eval()
    running_loss = 0.0

    with torch.no_grad():
        for iteration, (inputs, Y, _, _) in enumerate(test_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float()

            # sinogram去噪，noise2noise训练
            x1_denoised = x1
            x2_denoised = model_pre(x2)
            # x2_denoised = x2
            # 平均输出的sinogram
            aver_x = x2_denoised
            aver_x1 = model_recon(aver_x)
            mid_recon_t = normalization2one(radon.backprojection(radon.filter_sinogram(aver_x)))
            mid_recon_list = mid_recon_t if iteration == 0 else torch.cat([mid_recon_list, mid_recon_t], dim=0)
            label_list = Y if iteration == 0 else torch.cat([label_list, Y], dim=0)
        mid_recon_list = mid_recon_list.squeeze()
        label_list = label_list.squeeze()
        psnr_l, ssim_l = calculate_metrics(mid_recon_list, label_list)
        psnr_avg = psnr_l.mean()
        ssim_avg = ssim_l.mean()
        logger.info(f'psnr: {psnr_avg}, ssim: {ssim_avg}')
        return psnr_avg, ssim_avg


def main(args, writer):
    # # 数据
    from model.network_swinTrans import SwinIR

    file_path = os.path.join(args.root_path, f'angular_{args.n_theta}')
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank_t = torch.device("cpu")
    # radon = Radon(n_theta, circle=True, device=rank)
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    radon = Radon(resolution=128, angles=angles, clip_to_circle=True)
    dataset = DatasetPETRecon(file_path, radon, args.ratio, mode=args.mode, scale_factor=args.scale_factor)
    # radon = radon.to(rank)
    if args.fast:
        dataset = torch.utils.data.Subset(dataset, range(10))

    # 将数据集按80/10/10比例划分为训练集、验证集和测试集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 使用DataLoader加载数据
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 模型初始化

    denoise_model_pre = SwinIR(upscale=1,
                               in_chans=1,
                               img_size=[180, 128],
                               window_size=4,
                               patch_size=[45, 1] if args.aod else [1, 32],
                               img_range=1.0,
                               depths=[2, 6, 2],
                               embed_dim=180,
                               num_heads=[3, 6, 12],
                               mlp_ratio=2.0,
                               upsampler='',
                               resi_connection='1conv',
                               window_size_plus=1).to(rank)
    # denoise_model_pre = PETDenoiseNet(device=rank).to(rank)
    # log_writer.add_graph(denoise_model_pre, torch.randn(args.bs, 1, 128, 180).to(rank))
    denoise_model = WaveletFilterNet(180, 128).to(rank)

    # log_writer.add_graph(denoise_model, [torch.randn(args.bs, 1, 128, 128).to(rank), torch.randn(args.bs, 1, 128, 180).to(rank), torch.randn(args.bs, 1, 128, 180).to(rank)])

    logger = log_gen(args)
    if args.checkpoints:
        denoise_model_pre.load_state_dict(torch.load(f'./log/log_file_8160963/denoise_pre_weight_best.pth'))
        logger.info('load pre model...')
        # denoise_model.load_state_dict(torch.load(f'./model/denoise_weight_best.pth'))
    # print(torch.cuda.memory_summary())
    denoise_model_pre = nn.DataParallel(denoise_model_pre)
    denoise_model = nn.DataParallel(denoise_model)
    logger.info(denoise_model_pre)
    logger.info(denoise_model)
    if args.loss == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # criterion = nn.CrossEntropyLoss()
    # 创建优化器，为不同的网络设置不同的学习率
    optimizer = optim.SGD([
        {'params': denoise_model_pre.parameters(), 'lr': args.lr},  # net1 的学习率为 0.01
        {'params': denoise_model.parameters(), 'lr': 10 * args.lr}  # net2 的学习率为 0.001
    ])

    # 训练
    num_epochs = 40
    val_loss_best = 1
    if not args.test:
        for epoch in range(num_epochs):
            logger.info('start train !')
            logger.info(f'{5 * "*"}Epoch {epoch + 1}/{num_epochs}{5 * "*"}')
            train_loss = train(denoise_model_pre, denoise_model, radon, train_loader, criterion, optimizer, rank, epoch,
                               logger, writer)
            logger.info('start validate !')
            val_loss = validate(denoise_model_pre, denoise_model, radon, val_loader, criterion, rank, epoch, logger,
                                writer)
            logger.info(f'Epoch {epoch + 1}/{num_epochs} done! Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            writer.close()  # 训练结束，不再写入数据，关闭writer
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                torch.save(denoise_model_pre.state_dict(), os.path.join(args.log_dir, "denoise_pre_weight_best.pth"))
                torch.save(denoise_model.state_dict(), os.path.join(args.log_dir, "./denoise_weight_best.pth"))
                logger.info(f'Model saved! best in {epoch} for {val_loss:.4f}')
    logger.info('start test !')
    psnr, ssim = test(denoise_model_pre, denoise_model, radon, test_loader, criterion, rank, logger, writer)
    logger.info(f'Test Loss: psnr={psnr:.4f}, ssim={ssim:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PET reconstruction demo')
    parser.add_argument('--n_theta', default=180, type=int, help='number of theta')
    parser.add_argument('--bs', default=4, type=int, help='batch_size')
    parser.add_argument('--root_path', default='./simulation_angular/', type=str,
                        help='Input images')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--ratio', default=0.8, type=float, help='not noise ratio')

    parser.add_argument('--alpha', default=1, type=float, help='importance of the smooth of sinogram')
    parser.add_argument('--beta', default=1, type=float, help='importance of the delta of pic')
    parser.add_argument('--log_dir', default='./log_file/', type=str,
                        help='Directory for results')
    parser.add_argument('--weights',
                        default='./model/', type=str,
                        help='Path to weights')
    parser.add_argument('--mode', default='none', type=str, help='mode of noise')
    parser.add_argument('--scale_factor', default=0.5, type=float, help='counts level for poisson noise, HD*0.5=LD')
    parser.add_argument('--aod', default=True, type=bool,
                        help='patch of the angular or distance, the former by default')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='4', type=str, help='number of CUDA_VISIBLE_DEVICES')
    parser.add_argument('--opt_path', default='./modelSwinUnet/training.yaml', type=str,
                        help='path of SwinUnet preset file')
    parser.add_argument('--loss', default='L1', type=str, help='loss mode, L1 or L2')
    parser.add_argument('--show_tr', default=False, type=bool, help='whether to show results')
    parser.add_argument('--show_val', default=True, type=bool, help='whether to show results')
    parser.add_argument('--checkpoints', default=False, type=bool, help='whether to continue the last training')
    parser.add_argument('--test', default=False, type=bool, help='only for test or not')
    parser.add_argument('--fast', default=True, type=bool, help='small samples just for debug')

    args = parser.parse_args()
    seed = random.randint(0, 10000000)
    # seed = 8238551
    preset_seed(seed)
    n_theta = 180
    recon_size = 128
    # 配置logging模块

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    # logger.info(opt)
    # 初始化 SummaryWriter
    writer_1 = SummaryWriter(log_dir=os.path.join(args.log_dir, f"experiment_{seed}"))  # 可以指定日志路径

    main(args, writer_1)
