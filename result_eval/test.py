import sys
import os

import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from evaluate import calculate_metrics, plot_box
from utils.data import DatasetPETRecon
from utils.normalize import normalization2one
from utils.radon import Radon
from model.network_swinTrans import SwinIR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '/')))
from train1 import validate
from model.whole_network import PETReconNet, PETDenoiseNet


def eval_test(model_pre, radon, val_loader, rank):
    model_pre.eval()

    with torch.no_grad():
        for iteration, (inputs, Y, sino_label, picLD) in enumerate(val_loader):
            x1, x2 = inputs
            x1, x2 = x1.to(rank), x2.to(rank)
            Y = Y.to(rank).float().squeeze()
            picLD = picLD.to(rank)
            sino_label = sino_label.to(rank)

            # sinogram去噪，noise2noise训练
            x1_denoised = x1
            x2_denoised = model_pre(x2)
            # x2_denoised = x2
            # 平均输出的sinogram
            sino_recon = (x1_denoised + normalization2one(x2_denoised)) / 2
            pic_recon = normalization2one(radon.filter_backprojection(sino_recon))

            # PET图去噪
            sino_recon_list = sino_recon if iteration==0 else torch.cat([sino_recon_list, sino_recon], dim=0)
            pic_recon_list = pic_recon if iteration==0 else torch.cat([pic_recon_list, pic_recon], dim=0)
            picLD_list = picLD if iteration==0 else torch.cat([picLD_list, picLD], dim=0)
            sino_label_list = sino_label if iteration==0 else torch.cat([sino_label_list, sino_label], dim=0)
            picHD_list = Y if iteration==0 else torch.cat([picHD_list, Y], dim=0)

        return pic_recon_list.squeeze(), picLD_list.squeeze(), picHD_list.squeeze(), sino_label_list.squeeze(), sino_recon_list.squeeze()


if __name__ == '__main__':
    with torch.no_grad():
        # 数据导入
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 6"
        radon_me = Radon(n_theta=180, circle=True, device='cuda')
        test_dataset = DatasetPETRecon(file_path='../simulation_angular/angular_180',
                                       radon=radon_me, ratio=0.2, name_pre='transverse')
        all_in, all_label = test_dataset.get_all_in()
        all_in, all_label = all_in[0:200].to('cuda'), all_label[0:200].to('cuda')
        radon_pic_recon = radon_me.filter_backprojection(all_in)
        test_dataset = Subset(test_dataset, range(200))
        test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
        # 模型导入

        # with open('../modelSwinUnet/training.yaml', 'r') as config:
        #     opt = yaml.safe_load(config)
        device = 'cuda'
        model_pre = SwinIR(upscale=1,
                               in_chans=1,
                               img_size=[128, 180],
                               window_size=4,
                               patch_size=[1, 45],
                               img_range=1.0,
                               depths=[2, 6, 2],
                               embed_dim=180,
                               num_heads=[3, 6, 12],
                               mlp_ratio=2.0,
                               upsampler='',
                               resi_connection='1conv', )
        # model_pre = nn.DataParallel(model_pre).to(device)
        # model_recon = PETReconNet(radon_me, device, opt).to(device)
        model_pre.load_state_dict(torch.load('../model/denoise_pre_weight_bad.pth'))
        model_pre = model_pre.to(device)
        pic_recon_list, picLD_list, picHD_list, sino_label_list, sino_recon_list = eval_test(model_pre, radon_me, test_loader, device)
        pic_me_psnr, pic_me_ssim = calculate_metrics(pic_recon_list, picHD_list)
        pic_radon_psnr, pic_radon_ssim = calculate_metrics(radon_pic_recon.squeeze(), picHD_list)
        for i in range(10):
            plt.imshow(pic_recon_list[i].cpu().numpy()), plt.title('me_recon'), plt.show()
            plt.imshow(radon_pic_recon.squeeze()[i].cpu().numpy()), plt.title('radon_recon'), plt.show()
            plt.imshow(picHD_list.squeeze()[i].cpu().numpy()), plt.title('pic_HD'), plt.show()
            plt.imshow(picLD_list.squeeze()[i].cpu().numpy()), plt.title('pic_LD'), plt.show()
            plt.imshow(sino_label_list.squeeze()[i].cpu().numpy()), plt.title('sino_label'), plt.show()
            plt.imshow(sino_recon_list.squeeze()[i].cpu().numpy()), plt.title('sino_recon'), plt.show()
        data_list = [pic_me_psnr, pic_radon_psnr]
        label_list = ['me', 'radon']
        plot_box(data_list, label_list, 'PSNR (dB)', 'PSNR')
        data_list = [pic_me_ssim, pic_radon_ssim]
        plot_box(data_list, label_list, 'SSIM', 'SSIM')
        sino_psnr_l, sino_ssim_l = calculate_metrics(sino_recon_list, sino_recon_list)

