"""
Created on May 2019
PET image reconstruction demo


@author: Abi Mehranian
abolfazl.mehranian@kcl.ac.uk

"""
import numpy as np
from matplotlib import pyplot as plt
from geometry.BuildGeometry_v4 import BuildGeometry_v4
from utils.transfer_si import i2s
import scipy.io as sio
# from phantoms.brainweb import PETbrainWebPhantom

# temPath = r'C:\pythonWorkSpace\tmp'
geoPath = r'D:\pregraduate\PETrecon\tmp_180_128128'
# phanPath = r'E:\PET-M\Phantoms\Brainweb'

radialBinCropFactor = 0
PET = BuildGeometry_v4('mmr',radialBinCropFactor)
PET.loadSystemMatrix(geoPath,is3d=False)

AN = np.load('simulation_angular/angular_180/sino_AN.npy', allow_pickle=True)
pic = np.load('simulation_angular/angular_180/pic_HD.npy', allow_pickle=True)
# 创建正弦图
# theta = np.linspace(0., 360., max(sinogram.shape), endpoint=False)

# sino_me = i2s(pic[2, :, :], AN[2, :, :])
# img_3d_batch, mumap_3d_batch, t1_3d_batch, _ = PETbrainWebPhantom(phanPath, phantom_number=[0,2,10], voxel_size= np.array(PET.image.voxelSizeCm)*10, \
#                                            image_size=PET.image.matrixSize, pet_lesion = False, t1_lesion = False)
# 2D PET --------------------------------------------------


# img_2d = img_3d_batch[0,:,:,50]
# mumap_2d = mumap_3d_batch[0,:,:,50]
# img_2d_batch = img_3d_batch[:,:,:,50]
# mumap_2d_batch = mumap_3d_batch[:,:,:,50]
psf_cm = 0.4

## 2D forward project
#y = PET.forwardProjectBatch2D(img_2d, psf = psf_cm)
#y_batch = PET.forwardProjectBatch2D(img_2d_batch, psf = psf_cm)
path = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat\100_070722_1_20220707_162729_0.mat"
path2 = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat\100_070722_1_20220707_162729_1.mat"

data1 = sio.loadmat(path)['img'][:, 0:128, :]
data2 = sio.loadmat(path2)['img'][:, 0:128, :]
data = np.concatenate((data1, data2), axis=0)
# simulate 2D noisy sinograms
data = np.repeat(data, 16, axis=0)
sino_pet,AF,_ = PET.simulateSinogramData(data, counts= 1e7, psf = 0)
# plt.imshow(sino_me, 'gray')
# plt.show()
plt.imshow(sino_pet[2, :, :], 'gray')
plt.show()


# y_batch,AF_batch,_ = PET.simulateSinogramData(img_2d_batch, mumap = mumap_2d_batch, counts= 1e6,  psf = psf_cm)
#
# # 2D OSEM
# img_osem_2d = PET.OSEM2D(y, AN=AF, niter = 10, nsubs = 6, psf= 0.2)
# img_osem_2d_batch = PET.OSEM2D(y_batch, AN=AF_batch, niter = 10, nsubs = 6, psf= 0.2)







