import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import os

from torch import nn

from filter_design.wavelet_trans import WaveletFilterNet
from model.network_swinTrans import SwinIR
from model.whole_network import PETReconNet, PETDenoiseNet
# from utils.radon import Radon
from torch_radon import Radon as MeRadon
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 7"
data_test = np.load('./simulation_angular/angular_180/test_transverse_sinoLD.npy', allow_pickle=True)[:3, ]
pic_test = np.load('./simulation_angular/angular_180/test_transverse_picHD.npy', allow_pickle=True)[:3, ]

with open('./modelSwinUnet/training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

angles = np.linspace(0, np.pi, 180, endpoint=False)
radon = MeRadon(resolution=128, angles=angles, clip_to_circle=True)

with torch.no_grad():
    data_test = torch.from_numpy(data_test).to('cuda').float()
    data_test = data_test.unsqueeze(1)
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
                               resi_connection='1conv', ).to('cuda')
    # model_pre = PETDenoiseNet(device='cuda')
    model_pre = nn.DataParallel(model_pre)
    model_recon = WaveletFilterNet(180, 128).to('cuda')
    model_recon = nn.DataParallel(model_recon)
    model_pre.load_state_dict(torch.load('/home/ssd/linshuijin/PETrecon_backup/log/log_file_579842/denoise_pre_weight_best.pth'))
    model_recon.load_state_dict(torch.load('/home/ssd/linshuijin/PETrecon_backup/log/log_file_579842/denoise_weight_best.pth'))

    data_test = model_pre(data_test)
    data_test1 = model_recon(data_test.transpose(-1, -2))
    mask = torch.ones_like(data_test)

    pic = radon.backprojection(radon.filter_sinogram(data_test.transpose(-1, -2)))
    pic1 = radon.backprojection(radon.filter_sinogram(data_test1))
    # radon_me.filter_backprojection(data_test)
    plt.imshow(data_test.transpose(-1, -2).squeeze().cpu().numpy()[0, :, :])
    plt.show()
    plt.imshow(data_test1.squeeze().cpu().numpy()[0, :, :])
    plt.show()
    # pic1 = radon_me.filter_backprojection(data_test_1)
    # data_recon = model_recon(pic, data_test, mask)
    plt.imshow(pic.squeeze().cpu().numpy()[0, :, :])
    plt.show()
    plt.imshow(pic1.squeeze().cpu().numpy()[0, :, :])
    plt.show()


def visulize(data, pic):
    plt.imshow(data[0, 0, :, :].cpu().numpy())
    plt.show()
    plt.imshow(pic[0, 0, :, :])
    plt.show()
