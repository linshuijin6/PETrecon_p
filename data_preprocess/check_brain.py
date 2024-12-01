import os
import numpy as np
import scipy.io as sio
import pydicom
from skimage.util import img_as_float
from collections import defaultdict

# 输入和输出文件夹路径
input_folder = r"E:\dataset_pet\UDPET_Brain\dataset\dataset\train_mat"  # 替换为你的路径
output_main_folder = r"E:\dataset_pet\UDPET_Brain\IMA_check"  # 替换为你的输出路径

# 创建输出主文件夹
if not os.path.exists(output_main_folder):
    os.makedirs(output_main_folder)

# 获取所有文件
files = sorted(os.listdir(input_folder))

# 字典用于将同一受试者的文件归类
subject_files = defaultdict(list)

# 按照文件名前缀将每个受试者的文件分组
for filename in files:
    if filename.endswith('.mat'):
        # 通过文件名的前部分（例如 '100_070722_1'）确定受试者
        subject_id = '_'.join(filename.split('_')[:3])
        subject_files[subject_id].append(filename)

# 遍历每个受试者的文件列表
for subject_id, mat_files in subject_files.items():
    # 为每个受试者创建输出文件夹
    output_folder = os.path.join(output_main_folder, subject_id)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    slice_counter = 0  # 用于计算切片数

    # 遍历每个受试者的 .mat 文件
    for mat_file in mat_files:
        mat_file_path = os.path.join(input_folder, mat_file)

        # 加载 .mat 文件
        data = sio.loadmat(mat_file_path)
        image = data['img']  # 假设数据键为 'img'

        # 提取低剂量 PET 图像
        # image 维度为 [1, 256, 128]，我们取 [:, 0:128, :]
        low_dose_pet = image[:, 0:128, :]  # 现在的维度是 [1, 128, 128]

        # 生成 IMA 文件
        num_slices = low_dose_pet.shape[2]  # 第三维度的 128 为切片数
        for i in range(num_slices):
            slice_data = low_dose_pet[0, :, i]  # 获取每一张 2D 切片

            # 生成 DICOM 文件 (IMA)
            ds = pydicom.Dataset()
            ds.PixelData = img_as_float(slice_data).tobytes()
            ds.Rows, ds.Columns = slice_data.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16  # 假设使用16位深度
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            # 保存为 IMA 文件
            dicom_filename = os.path.join(output_folder, f'slice_{slice_counter + 1:04d}.IMA')
            pydicom.dcmwrite(dicom_filename, ds)
            slice_counter += 1
            print('已保存：', dicom_filename)

print("所有文件已处理并保存为 IMA 格式。")
