import time

import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon


# 假设您已经有了小波变换后的频率成分，例如 ndct_LL, ldct_LL 等
# 这里的代码以 ndct_LL 和 ldct_LL 为例进行演示
def plot_image_and_histogram(t1, t2, ndct_LL, ldct_LL):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 左侧显示小波分量图像
    axes[0, 0].imshow(ndct_LL, cmap='gray')
    axes[0, 0].set_title(t1)
    axes[0, 1].imshow(ldct_LL, cmap='gray')
    axes[0, 1].set_title(t2)

    ndct_LL = ndct_LL.flatten()
    # ndct_LL = ndct_LL[ndct_LL > 0]
    #
    ldct_LL = ldct_LL.flatten()
    # ldct_LL = ldct_LL[ldct_LL > 0]
    cos_sim = cosine_similarity(ndct_LL.reshape(1, -1), ldct_LL.reshape(1, -1))[0][0]  # 余弦相似度, 1表示完全相似，-1表示完全不同， 0表示无关

    # 将数据归一化为概率分布（可以通过直方图来做）
    hist1, _ = np.histogram(ldct_LL, bins=30, range=(0, 1), density=True)
    hist2, _ = np.histogram(ndct_LL, bins=30, range=(0, 1), density=True)

    # 计算 J-S 散度
    jsd = jensenshannon(hist1, hist2) # J-S 散度, 越小越相似， 0~1

    # 计算直方图并在右侧绘制
    axes[1, 0].hist(ndct_LL, bins=50, color='blue', alpha=0.5, label=t1)
    axes[1, 0].hist(ldct_LL, bins=50, color='green', alpha=0.5, label=t2)
    axes[1, 0].set_title('Histogram, J-S={:.4f}, cos_sim={:.4f}'.format(jsd, cos_sim))

    # 如果需要，显示更高频的分量直方图
    # axes[1, 2].hist(np.abs(ndct_HL_norm - ldct_HL_norm).flatten(), bins=50, color='red', alpha=0.7)
    # axes[1, 2].set_title('Difference Histogram')
    # 添加图例
    plt.legend()

    plt.tight_layout()
    plt.show()


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


# 假设已有NDCT和LDCT图像，存储在ndct_img和ldct_img变量中
def perform_dwt(img):
    # 这里使用Haar小波（'haar'）做简单的小波变换，您可以根据需要选择其他小波
    coeffs = pywt.dwt2(img, 'haar')  # 二维小波变换
    LL, (LH, HL, HH) = coeffs
    # recon = pywt.idwt2(coeffs, 'haar')  # 二维小波逆变换
    # plt.imshow(recon, cmap='gray')
    # plt.show()
    # r = (img - recon).flatten().sum()
    # print(r)
    return LL, LH, HL, HH


# ldct_img = np.load('../simulation_angular/angular_180/test_transverse_picLD.npy', allow_pickle=True)[0, ].squeeze()
# ndct_img = np.load('../simulation_angular/angular_180/test_transverse_picHD.npy', allow_pickle=True)[0, ].squeeze()
if __name__ == '__main__':
    for i in range(6):
        ldct_sino = np.load('../simulation_angular/angular_180/test_transverse_sinoLD.npy', allow_pickle=True)[i, ].squeeze()
        ndct_sino = np.load('../simulation_angular/angular_180/test_transverse_sinoHD.npy', allow_pickle=True)[i, ].squeeze()
        plt.imshow(ndct_sino, 'gray')
        plt.show()

        # 计算NDCT和LDCT的DWT
        ndct_LL, ndct_LH, ndct_HL, ndct_HH = perform_dwt(ndct_sino)
        ldct_LL, ldct_LH, ldct_HL, ldct_HH = perform_dwt(ldct_sino)

        ndct_LL = normalize(ndct_LL)
        ldct_LL = normalize(ldct_LL)

        ndct_LH = normalize(ndct_LH)
        ldct_LH = normalize(ldct_LH)

        # 其他高频分量的归一化
        ndct_HL = normalize(ndct_HL)
        ldct_HL = normalize(ldct_HL)
        ndct_HH = normalize(ndct_HH)
        ldct_HH = normalize(ldct_HH)

        fig, ax = plt.subplots(2, 3, figsize=(12, 8))

        # 低频分量
        ax[0, 0].imshow(ndct_LL, cmap='gray')
        ax[0, 0].set_title('NDCT LL')
        ax[0, 1].imshow(ldct_LL, cmap='gray')
        ax[0, 1].set_title('LDCT LL')

        # 高频分量
        ax[1, 0].imshow(ndct_LH, cmap='gray')
        ax[1, 0].set_title('NDCT LH')
        ax[1, 1].imshow(ldct_LH, cmap='gray')
        ax[1, 1].set_title('LDCT LH')

        ax[1, 2].imshow(np.abs(ndct_HL - ldct_HL), cmap='hot')
        ax[1, 2].set_title('Difference in HL')

        plt.show()

        n_full = np.concatenate((ndct_HL, ndct_HH, ndct_LH), axis=0)
        l_full = np.concatenate((ldct_HL, ldct_HH, ldct_LH), axis=0)
        # 假设您已经计算了NDCT和LDCT的小波分量（如ndct_LL, ldct_LL等）
        plot_image_and_histogram('ndct_LL', 'ldct_LL', ndct_LL, ldct_LL)
        plot_image_and_histogram('ndct_HH', 'ldct_HH', ndct_HH, ldct_HH)
        plot_image_and_histogram('ndct_LH', 'ldct_LH', ndct_LH, ldct_LH)
        plot_image_and_histogram('ndct_HL', 'ldct_HL', ndct_HL, ldct_HL)
        plot_image_and_histogram('ndct_full', 'ldct_full', n_full, l_full)
        time.sleep(2)
