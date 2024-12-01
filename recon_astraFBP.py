import astra
import numpy as np
import torch
from matplotlib import pyplot as plt


def sino2pic(sino, pic_size=168, filter='hamming'):
    assert sino.shape[-1] % 180 == 0
    theta = sino.shape[-1]
    # 定义投影几何和图像几何
    sino = sino.T.squeeze()
    sino = sino.cpu().numpy() if isinstance(sino, torch.Tensor) else sino
    proj_geom = astra.create_proj_geom('parallel', 1.0, pic_size, np.linspace(0, np.pi, theta))
    vol_geom = astra.create_vol_geom(pic_size, pic_size)

    projector_id = astra.create_projector('cuda', proj_geom, vol_geom)

    # 创建投影数据（正弦图）
    # sinogram_data = np.load(r'E:\project_pet\FBSEM-master\simulation_angular\angular_360\transverse\sino_HD.npy',
    #                         allow_pickle=True)[2, :, :].transpose(1, 0)

    sinogram_id = astra.data2d.create('-sino', proj_geom, sino)

    # FBP 重建
    reconstruction_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectorId'] = projector_id
    cfg['ReconstructionDataId'] = reconstruction_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {}
    cfg['option']['FilterType'] = filter
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # 获取重建结果
    x = astra.data2d.get(reconstruction_id)
    x = (x - x.min()) / (x.max() - x.min())

    # plt.imshow(x, cmap='gray')
    # plt.show()

    # 清理
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sinogram_id)
    astra.data2d.delete(reconstruction_id)
    return torch.from_numpy(x)
