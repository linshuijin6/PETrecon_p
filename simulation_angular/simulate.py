from phantoms.brainweb import PETbrainWebPhantom
import numpy as np
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from angular_miss import generate_mask
import matplotlib.pyplot as plt
pic = np.load('./angular_180/pic_HD.npy', allow_pickle=True)
AN = np.load('./angular_360/sino_AN.npy', allow_pickle=True)
sino = np.load('./angular_360/sino_HD.npy', allow_pickle=True)
sigma_g = [0, 0.05, 0.075, 0.0875, 0.1, 0.2, 0.4, 0.6]
for sigma in sigma_g:
    # 使用或不使用AN（衰减校正）得到的重建结果比较
    AN_F = np.ones_like(AN)
    mask_p1, mask_p2 = generate_mask(sino.shape, sigma)
    sino_r = mask_p1*sino
    temPath = r'./tmp_1'
    PET = BuildGeometry_v4('mmr', 0.5)  #scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    img_hd = PET.OSEM2D(sino, AN=AN, niter=10, nsubs=6, psf=0.25)
    # img_hd_noAN = PET.OSEM2D(sino, AN=AN_F, niter=10, nsubs=6, psf=0.25)
    # plt.imshow(img_hd[5, :, :], cmap='gray'), plt.title('AN'), plt.show()
    # plt.imshow(img_hd_noAN[5, :, :], cmap='gray'), plt.title('noAN'), plt.show()
    plt.imshow(sino_r[5, :, :], cmap='gray', interpolation='none'), plt.title(f'sigma = {sigma}'), plt.axis('off'), plt.savefig(f'./angular_360/sino_sigma={sigma}.png', bbox_inches='tight', pad_inches=0), plt.close()
    plt.imshow(img_hd[5, :, :], cmap='gray', interpolation='none'), plt.title(f'sigma = {sigma}'), plt.axis('off'), plt.savefig(f'./angular_360/recon_sigma={sigma}.png', bbox_inches='tight', pad_inches=0), plt.close()
    1