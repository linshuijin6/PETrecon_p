import numpy as np
import matplotlib.pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from models.deeplib import buildBrainPhantomDataset
from phantoms.brainweb import PETbrainWebPhantom


def generate_mask(dimensions, sigma):
    """
    生成batchsize个mask，对应的列置为0。

    参数:
    dimensions: tuple，图像尺寸，格式为(batchsize, radical, angular)
    sigma: float，置0的列占总列数的比值。

    输出:
    mask: np.array, 尺寸与输入尺寸一致的mask。
    """
    batchsize, radical, angular = dimensions
    # 初始化mask为全1
    mask = np.ones((batchsize, radical, angular))

    # 计算每个batch中需要置0的列数
    num_zero_columns = int(sigma * angular)

    for i in range(batchsize):
        # 随机选择需要置0的列索引
        zero_columns = np.random.choice(angular, num_zero_columns, replace=False)
        # 将对应列的值置为0
        mask[i, :, zero_columns] = 0

    mask_p1 = mask
    mask_p2 = np.ones_like(mask)-mask

    return mask_p1, mask_p2


if __name__ == '__main__':
    temPath = r'./tmp_1'
    PET = BuildGeometry_v4('mmr', 0)  #scanner mmr, with radial crop factor of 50%
    PET.loadSystemMatrix(temPath, is3d=False)
    # PET.plotMichelogram()

    phanPath = r'E:\dataset_pet\BrainWeb\raw'
    save_training_dir = r'./data_aug/tra-t'
    phanType = 'brainweb'
    phanNumber = np.arange(0, 1, 1)  # use first 5 brainweb phantoms out of 20
    voxel_size = np.array(PET.image.voxelSizeCm) * 10
    image_size = PET.image.matrixSize
    img, mumap, _, _ = PETbrainWebPhantom(phanPath, 0, voxel_size, image_size, 0, 0, False,
                                          False, False, 0.1)
    print("* simulate HD 2D sinograms...")
    slice_index = np.arange(65,85,2)
    img = np.transpose(img[:, :, slice_index], (2, 0, 1))
    img[img < 0] = 0
    np.save('../simulation_angular/label.npy', img)
    mumap = np.transpose(mumap[:, :, slice_index], (2, 0, 1))
    mumap[mumap < 0] = 0
    assert img.shape[1] == img.shape[2]  # 确保img的第一个维度为batch或slice，否则下行的函数报错
    prompts_hd, AF, NF, _ = PET.simulateSinogramData(img, mumap=mumap, counts=1e7, psf=0.25)
    np.save('../simulation_angular/angular_360/sino_HD.npy', prompts_hd)
    AN = AF * NF
    np.save('../simulation_angular/angular_360/sino_AN.npy', AN)
    img_hd = PET.OSEM2D(prompts_hd, AN=AN, niter=10, nsubs=6, psf=0.25)
    np.save('../simulation_angular/angular_360/pic_HD.npy', img_hd)

    prompts_hd, AF, NF, _ = PET.simulateSinogramData(img, mumap=mumap, counts=1e7, psf=0.25)
    AN = AF * NF
    img_hd = PET.OSEM2D(prompts_hd, AN=AN, niter=10, nsubs=6, psf=0.25)
    buildBrainPhantomDataset(PET, save_training_dir, phanPath, counts_hd=5e6, count_ld_window_2d=[5e5, 7.5e5],
                             phanType=phanType, phanNumber=phanNumber, is3d=False, pet_lesion=False, num_rand_rotations=3,
                             niter_hd=10, nsubs_hd=6, rot_angle_degrees=15)
