import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 输入文件夹路径
input_folder = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat"  # 存放.mat文件的文件夹
output_folder = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_png"  # 存放.png文件的文件夹
# 批量处理所有.mat文件
for mat_file in os.listdir(input_folder):
    if mat_file.endswith('.mat'):
        # 构造文件路径
        mat_path = os.path.join(input_folder, mat_file)

        # 读取.mat文件
        mat_data = loadmat(mat_path)

        # 假设.mat文件中的数据变量名是'array'
        if 'img' in mat_data:
            data = mat_data['img']

            # 提取指定的子数组 [1, 128:256, :]
            data_slice = data[0, 128:, :]  # 提取 [1, 128:256, :] 的部分

            # 显示数据切片
            plt.imshow(data_slice, cmap='gray')
            plt.title(f'File: {mat_file}')
            # 保存图片为png格式
            output_path = os.path.join(output_folder, f'{mat_file}.jpg')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='jpg')  # 指定格式为jpg
            plt.close()  # 关闭当前绘图，防止占用内存

