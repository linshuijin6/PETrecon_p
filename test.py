import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import os

from torch import nn
from torch.utils.data import DataLoader

from filter_design.wavelet_trans import WaveletFilterNet
from model.network_swinTrans import SwinIR
from model.whole_network import PETReconNet, PETDenoiseNet
# from utils.radon import Radon
from torch_radon import Radon as MeRadon

from result_eval.evaluate import calculate_metrics
from utils.data import DatasetPETRecon


def test(model_pre, radon, test_loader, rank):
    model_pre = nn.DataParallel(model_pre)
    model_pre.load_state_dict(torch.load(os.path.join('./log/log_file_8837470', "denoise_pre_weight_best.pth")))
    model_pre.eval()
    with torch.no_grad():
        for iteration, (inputs, Y, _, _) in enumerate(test_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float()

            x2_denoised = model_pre(x2)
            mid_recon_t = radon.backprojection(radon.filter_sinogram(x2_denoised))
            mid_recon_list = mid_recon_t if iteration == 0 else torch.cat([mid_recon_list, mid_recon_t], dim=0)
            label_list = Y if iteration == 0 else torch.cat([label_list, Y], dim=0)
            print(f'iteration: {iteration}/{len(test_loader)} done! ')
        mid_recon_list = mid_recon_list.squeeze()
        label_list = label_list.squeeze()
        psnr_l, ssim_l = calculate_metrics(mid_recon_list, label_list)
        psnr_avg = psnr_l.mean()
        ssim_avg = ssim_l.mean()
        print(f'psnr: {psnr_avg}, ssim: {ssim_avg}')
        return psnr_avg, ssim_avg

def visulize(data, pic):
    plt.imshow(data[0, 0, :, :].cpu().numpy())
    plt.show()
    plt.imshow(pic[0, 0, :, :])
    plt.show()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # data_test = np.load('./simulation_angular/angular_180/test_transverse_sinoLD.npy', allow_pickle=True)
    # pic_test = np.load('./simulation_angular/angular_180/test_transverse_picHD.npy', allow_pickle=True).squeeze()

    file_path = './simulation_angular/angular_180'
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    radon = MeRadon(resolution=128, angles=angles, clip_to_circle=True)

    test_dataset = DatasetPETRecon(file_path, radon, 0.8, mode='none', scale_factor=2, name_pre='test_transverse')
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    rank = 'cuda:3'
    model_pre = SwinIR(upscale=1,
                               in_chans=1,
                               img_size=[180, 128],
                               window_size=4,
                               patch_size=[45, 1],
                               img_range=1.0,
                               depths=[2, 6, 2],
                               embed_dim=180,
                               num_heads=[3, 6, 12],
                               mlp_ratio=2.0,
                               upsampler='',
                               resi_connection='1conv', ).to(rank)
    psnr, ssim = test(model_pre, radon, test_loader, 'cuda:3')


