import os
import sys

import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '/')))
# from radon import Radon
# from normalize import normalization2one


# from recon_astraFBP import sino2pic as s2p


def normalization2one(input_tensor: torch.Tensor or np.ndarray) -> torch.Tensor:
    # 判断是否为 numpy 数组且维度为 (batchsize, h, w)，若是则转换为 tensor
    if isinstance(input_tensor, np.ndarray):
        assert len(input_tensor.shape) == 3, "NumPy input must have dimensions (batchsize, height, width)."
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(1).to('cuda')
    else:
        assert isinstance(input_tensor, torch.Tensor), "Input must be a PyTorch tensor or a NumPy array."
        assert len(input_tensor.shape) == 4, "Input tensor must have 4 dimensions (batchsize, channels, height, width)."

    input_tensor = input_tensor.contiguous()
    # 计算每个batch的最小值和最大值，避免reshape所带来的额外内存开销
    input_shape = input_tensor.shape
    min_val = input_tensor.view(input_tensor.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    max_val = input_tensor.view(input_tensor.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)

    # 进行归一化，将所有数值归一化到 [0, 1] 区间
    input_tensor = (input_tensor - min_val) / (max_val - min_val + 1e-8)  # 1e-8 防止除以0

    assert input_shape == input_tensor.shape
    # normalized_tensor = normalized_tensor * 1.0 - 0.5

    return input_tensor  # 确认输出形状 (batchsize, 1, h, w)

def tv_loss(img):
    # 假设 img 的形状为 (batch_size, channel=1, height, width)
    # 因为 channel 为 1，可以直接在高度和宽度维度上计算总变差

    # 计算 X 方向的总变差 (沿着宽度方向的差异)
    n_pixel = img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3]

    tv_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    n_tv_x = tv_x.shape[0] * tv_x.shape[1] * tv_x.shape[2] * tv_x.shape[3]
    v_tv_x = torch.sum(tv_x)

    # 计算 Y 方向的总变差 (沿着高度方向的差异)
    tv_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    n_tv_y = tv_y.shape[0] * tv_y.shape[1] * tv_y.shape[2] * tv_y.shape[3]
    v_tv_y = torch.sum(tv_y)

    mean_tv = (v_tv_x+v_tv_y)/(n_tv_x+n_tv_y)

    return mean_tv


def set_random_pixels_to_zero(data, ratio):
    """
    将输入张量中的部分像素值随机设为0.

    参数:
        data (torch.Tensor): 输入张量，形状为 (batchsize, c, h, w).
        ratio (float): 要设为0的像素占比（0到1之间的浮点数）。

    返回:
        torch.Tensor: 修改后的张量.
    """
    # 确保 ratio 在 [0, 1] 范围内
    if not (0 <= ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")

    batchsize, c, h, w = data.shape
    # 计算要设为0的像素数量
    num_pixels_to_zero = int(batchsize * c * h * w * ratio)

    # 随机选择要设为0的索引
    indices = torch.randperm(batchsize * c * h * w)[:num_pixels_to_zero]

    # 将对应的像素值设为0
    data_flat = data.view(-1)  # 展平张量
    data_flat[indices] = 0
    return data_flat.view(batchsize, c, h, w)  # 还原原始形状


# 加噪声函数（适用于 PyTorch tensor）
def add_noise(radon, img=None, sino=None, ratio=0.2, mode='none', scale_factor=0.5):
    """
    img: 输入的图像张量，假设值在 [0, 1] 范围内，形状为 [batch_size, channels, height, width]
    mode: 噪声类型 'poisson+gaussian' 或 'gaussian'
    gauss_mean: 高斯噪声的均值
    gauss_std: 高斯噪声的标准差
    """
    from .radon import Radon
    if mode == 'none':
        img = img.cuda().float()
        img = img[:, None, :, :] if len(img.shape)==3 else img
        noisy_image = img.clone().to('cuda')
        noisy_image = set_random_pixels_to_zero(noisy_image, ratio)
        # plt.imshow(noisy_image[0, 0, :, :].cpu().squeeze().numpy()), plt.show()
        me_radon = Radon(n_theta=180, circle=True, device='cuda')
        out_sino = normalization2one(me_radon(noisy_image.to('cuda')))
        del noisy_image
        out_sino = out_sino - ratio*normalization2one(sino.to('cuda'))
        return out_sino.squeeze(1)
    elif mode == 'poisson':
        sino = sino[:, None, :, :] if len(sino.shape) == 3 else sino
        t_sino = sino.clone()
        """
            对 PET 弦图数据添加泊松噪声。

            参数:
                sino (torch.Tensor): 输入的弦图数据，维度为 (batchsize, c, h, w)。
                scale_factor (float): 噪声控制因子。增大此值可以降低噪声的相对强度，减小此值可以提高噪声的相对强度。
                UDPET的数据中，LD的counts为8e4，HD的counts为15e4，约为两倍，于是设置scale_factor=0.5。

            返回:
                noisy_sino (torch.Tensor): 添加了泊松噪声的弦图数据。
            """
        # 将输入数据进行缩放
        inpu_sum = t_sino.view(-1).sum()/sino.size(0)
        inpu_level = [t_sino.view(-1).min(), t_sino.view(-1).max()]
        print(f'input sum: {inpu_sum}, input level: {inpu_level}')

        scaled_sino = t_sino * scale_factor

        # 转换为泊松分布所需的整数格式
        # 由于泊松分布只接受整数，确保数据非负
        scaled_sino = torch.clamp(scaled_sino, min=0)

        # 添加泊松噪声
        noised_sino = torch.poisson(scaled_sino)

        # 将数据缩放回原来的范围
        # noised_sino = noised_sino / scale_factor
        out_sum = noised_sino.view(-1).sum() / sino.size(0)
        out_level = [noised_sino.view(-1).min(), noised_sino.view(-1).max()]
        print(f'output sum: {out_sum}, output level: {out_level}')

        return noised_sino.squeeze(1).cpu()
        # for b in range(batch_size):
        #     scale_factor_randoms = scale_factor / noisy_sino[b, 0, :, :].sum()
        #     noisy_sino[b, 0, :, :] = torch.poisson(noisy_sino[b, 0, :, :] * scale_factor_randoms) + noisy_sino[b, 0, :, :]
        #
        # # noisy_sino = torch.poisson(noisy_image * scale_factor) / scale_factor  # 归一化回 [0, 1]
        # return noisy_sino.squeeze(1).cpu()


    #
    #
    # if mode == 'p+g':
    #     # 1. 泊松噪声：Poisson分布中的数值是整数，因此需要将图像值扩展为较大范围
    #     noisy_poisson = torch.poisson(img * 255.0) / 255.0  # 归一化回 [0, 1]
    #
    #     # 2. 高斯噪声：生成高斯噪声并加到泊松噪声图像上
    #     noise_gauss = torch.normal(mean=gauss_mean, std=gauss_std / 255.0, size=noisy_poisson.shape).to(img.device)
    #     noisy_gaussian = torch.clamp(noisy_poisson + noise_gauss, 0.0, 1.0)
    #
    #     return noisy_gaussian
    #
    # elif mode == 'g':
    #     # 仅添加高斯噪声
    #     noise_gauss = torch.normal(mean=gauss_mean, std=gauss_std, size=img.shape).to(img.device)
    #     noisy_gaussian = img + noise_gauss
    #     # noisy_gaussian = torch.clamp(img + noise_gauss, 0.0, 1.0)
    #
    #     return noisy_gaussian
    #
    # else:
    #     raise ValueError("Invalid noise mode. Choose 'p+g' or 'g'.")


def load_data(dir_path, name_pre):
    file_path_pre = dir_path + '/' + name_pre
    # file_path_pre = dir_path + '/' + name_pre
    file_sinoLD = np.load(file_path_pre + '_sinoLD.npy', allow_pickle=True)
    file_sinoHD = np.load(file_path_pre + '_sinoHD.npy', allow_pickle=True)
    file_imageLD = np.load(file_path_pre + '_picLD.npy', allow_pickle=True)
    file_imageHD = np.load(file_path_pre + '_picHD.npy', allow_pickle=True)

    # file_imageLD = np.rot90(file_imageLD, -1, (2, 3))
    # file_imageHD = np.rot90(file_imageHD, -1, (2, 3))

    # X_all = np.expand_dims(np.transpose(file_sinoLD, (0, 1, 2)), -1)
    # Y_all = np.expand_dims(np.transpose(file_imageHD, (0, 1, 2)), -1)
    X_all = file_sinoLD
    Y_all = file_imageHD


    return X_all, Y_all, file_imageLD, file_sinoHD


def generate_mask(dimensions, sigma, column=True):
    """
    生成batchsize个mask，对应的列置为0。

    参数:
    dimensions: tuple，图像尺寸，格式为(batchsize, radical, angular)
    sigma: float，置0的列占总列数的比值。

    输出:
    mask: np.array, 尺寸与输入尺寸一致的mask。
    """
    batchsize, _, radical, angular = dimensions
    # 初始化mask为全1
    mask = np.ones((batchsize, radical, angular))

    # 计算每个batch中需要置0的列数
    num_zero_columns = int(sigma * angular)

    for i in range(batchsize):
        # 随机选择需要置0的列索引
        zero_columns = np.random.choice(angular, num_zero_columns, replace=False)
        # 将对应列的值置为0
        if column:
            mask[i, :, zero_columns] = 0
        else:
            mask[i, zero_columns, :] = 0

    mask_p1 = mask
    mask_p2 = np.ones_like(mask) - mask

    return mask_p1, mask_p2


class DatasetPETRecon(data.Dataset):
    def __init__(self, file_path, radon, ratio, mode, name_pre='transverse', scale_factor=0.5):
        super().__init__()
        self.file_path = file_path
        self.radon = radon
        self.ratio = ratio
        self.name_pre = name_pre
        self.mode = mode
        self.scale_factor = scale_factor
        self.x1_noisy, self.x2_noisy, self.Y_train, self.sino_label, self.picLD_train = self.prep_data()

    def __getitem__(self, index):
        x1 = self.x1_noisy[index, :, :, :]
        x2 = self.x2_noisy[index, :, :, :]
        Y = self.Y_train[index, :, :, :]
        sino_label = self.sino_label[index, :, :]
        picLD_train = self.picLD_train[index, :, :, :]
        X = (x1, x2)
        # elif self.phase == 'test':
        #     X = self.X_test
        #     Y = self.Y_test
        # elif self.phase == 'val':
        #     X = self.X_val
        #     Y = self.Y_val
        return X, Y, sino_label, picLD_train

    def __len__(self):
        return self.Y_train.shape[0]

    def prep_data(self):
        file_path = self.file_path
        # 数据
        name_pre = self.name_pre
        # X, sinogram; Y, pic
        X_train, Y_train, picLD_train, sino_label = load_data(file_path, name_pre)
        X_train, Y_train, picLD_train = torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(picLD_train)
        # X_train_noisy1, X_train_noisy2 = add_noise(X_train, mode='g'), add_noise(X_train, mode='g')
        Y_train = Y_train.squeeze() if X_train.shape[0] != 1 else Y_train
        # picLD_train = picLD_train[:, None, :, :] if picLD_train.shape[1] != 1 else picLD_train
        # gau_std = torch.std(X_train).item()*0.1
        # X_train_noisy1, X_train_noisy2 = add_noise(picLD_train, self.radon, self.ratio, mode='poisson', scale_factor=8e4), X_train  # noise2noise策略
        X_train_noisy1, X_train_noisy2 = add_noise(self.radon, img=picLD_train, sino=X_train, ratio=self.ratio, mode=self.mode, scale_factor=self.scale_factor), X_train  # noise2noise策略
        X_train_noisy1 = torch.unsqueeze(X_train_noisy1, 1) if len(X_train_noisy1.shape) == 3 else X_train_noisy1
        X_train_noisy2 = torch.unsqueeze(X_train_noisy2, 1) if len(X_train_noisy2.shape) == 3 else X_train_noisy2
        Y_train = torch.unsqueeze(Y_train, 1) if len(Y_train.shape) == 3 else Y_train

        return X_train_noisy1.transpose(3, 2), X_train_noisy2.transpose(3, 2), Y_train, sino_label.transpose(0, 1, 3, 2), picLD_train

    def get_all_in(self):
        return self.x2_noisy, self.Y_train


if __name__ == '__main__':
    # 测试
    from radon import Radon
    # 加载数据
    root1_path = '/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_picLD.npy'
    root2_path = '/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_picHD.npy'
    root_path = '/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_sinoLD.npy'
    root3_path = '/home/ssddata/linshuijin/PETrecon/simulation_angular/angular_180/test_transverse_sinoHD.npy'
    file = np.load(root1_path, allow_pickle=True)[:4, :, :, :]
    file1 = np.load(root_path, allow_pickle=True)[:4, :, :]
    sino_HD = np.load(root3_path, allow_pickle=True)[:4, :, :]
    pic_HD = np.load(root2_path, allow_pickle=True)[:4, :, :, :]
    # bs = file.shape[0]
    # ave = file.reshape(bs, -1)
    # sum_l = ave.sum(axis=1)
    # a = sum_l.mean()
    device = 'cuda:2'
    picLD = torch.from_numpy(file).float().to(device)
    sinoLD = torch.from_numpy(file1).float().to(device).unsqueeze(1)
    picHD = torch.from_numpy(pic_HD).float().to(device)
    sinoHD = torch.from_numpy(sino_HD).float().to(device).unsqueeze(1)
    radon = Radon(n_theta=180, circle=True, device=device)
    plt.imshow(picHD[0, 0, :, :].cpu().numpy()), plt.title(f'pic_HD'), plt.show()
    plt.imshow(picLD[0, 0, :, :].cpu().numpy()), plt.title(f'pic_LD'), plt.show()
    for _ in range(10):
        dif_x = []
        dif_y = []
        min_v = 1
        r_m = 0.1
        for i in range(5, 10, 1):
            scale_factor = i/10
            sino_low = add_noise(radon, img=picLD, sino=sinoHD, ratio=0.4, mode='poisson', scale_factor=scale_factor)

            sino_low = sino_low.unsqueeze(1).to(device)
            plt.imshow(sino_low[0, 0, :, :].cpu().numpy()), plt.title(f'sino_low， scale_factor = {scale_factor}'), plt.show()
            pic_low = radon.filter_backprojection(sino_low)
            plt.imshow(pic_low[0, 0, :, :].cpu().numpy()), plt.title(f'scale_factor = {scale_factor}'), plt.show()
            dif_value = torch.nn.MSELoss()(normalization2one(pic_low), normalization2one(picLD)).item()
            dif_y.append(dif_value)
            # print(f'ratio={ratio}, dif={dif_value}')
            if dif_value < min_v:
                min_v = dif_value
                r_m = scale_factor
        print(f'scale_factor={r_m}, min_dif={min_v}')
        1




    # data = add_noise(radon, sino=file1, mode='poisson', scale_factor=1e4, ratio=0.4).unsqueeze(1).to('cuda:2')
    # data1, _ = add_noise(radon, img=file, sino=file1, mode='none', scale_factor=1e4, ratio=0.4)
    # data1 = data1.unsqueeze(1).to('cuda:2')
    # data2, noisy_pic = add_noise(radon, img=file2, sino=file3, mode='none', scale_factor=1e4, ratio=0.05)
    # data2 = data2.unsqueeze(1).to('cuda:2')
    # noisy_pic = noisy_pic.squeeze().to('cuda:2')
    # recon = radon.filter_backprojection(data).cpu().squeeze().numpy()
    # # data1 = normalization2one(data1) - 0.2*normalization2one(file1)
    # recon_1 = radon.filter_backprojection(data1).cpu().squeeze().numpy()
    # data = data.cpu().squeeze().numpy()
    # data1 = data1.cpu().squeeze().numpy()
    # data2 = data2.cpu().squeeze().numpy()
    # file = file.cpu().squeeze().numpy()
    # file1 = file1.cpu().squeeze().numpy()
    # noisy_pic = noisy_pic.cpu().squeeze().numpy()
    # # plt.imshow(data[0, :, :]), plt.show()
    # plt.imshow(data2[0, :, :]), plt.title('none_HD'), plt.show()
    # plt.imshow(file1[0, :, :]), plt.title('sino_LD'), plt.show()
    # plt.imshow(file[0, :, :]), plt.title('pic_LD'), plt.show()
    # # plt.imshow(recon[0, :, :]), plt.show()
    # # plt.imshow(recon_1[0, :, :]), plt.show()
    # plt.imshow(file1[0, :, :]-data2[0, :, :], cmap='gray'), plt.show()
    # plt.imshow(file[0, :, :]-noisy_pic[0, :, :], cmap='gray'), plt.show()
    # plt.imshow(pic_HD[0, :, :]), plt.show()
    # plt.imshow(sino_HD[0, :, :]), plt.show()



