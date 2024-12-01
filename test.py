import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import os
from model.whole_network import PETReconNet, PETDenoiseNet
from utils.radon import Radon
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_test = np.load('/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_sinoHD.npy', allow_pickle=True)
pic_test = np.load('/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_picHD.npy', allow_pickle=True)

with open('./modelSwinUnet/training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

radon_me = Radon(n_theta=180, circle=True, device='cuda')


with torch.no_grad():
    data_test = torch.from_numpy(data_test).to('cuda').float()
    data_test = data_test.unsqueeze(1)
    model_pre = PETDenoiseNet(device='cuda')
    model_recon = PETReconNet(device='cuda', radon=radon_me, config=opt)

    model_pre.load_state_dict(torch.load('./model/denoise_pre_weight.pth'))
    model_recon.load_state_dict(torch.load('./model/denoise_weight.pth'))

    data_test = model_pre(data_test)
    mask = torch.ones_like(data_test)

    pic = radon_me.filter_backprojection(data_test)
    data_recon = model_recon(pic, data_test, mask)
    plt.imshow(data_recon[0, 0, :, :].cpu().numpy())
    plt.show()
    plt.imshow(pic_test[0, 0, :, :])
    plt.show()


def visulize(data, pic):
    plt.imshow(data[0, 0, :, :].cpu().numpy())
    plt.show()
    plt.imshow(pic[0, 0, :, :])
    plt.show()
