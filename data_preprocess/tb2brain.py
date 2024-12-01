import os
import numpy as np
import scipy.io
import pydicom  # 用于读取 IMA 文件

# 文件夹路径
ima_folder = 'path/to/your/IMA/files'
output_folder = 'path/to/save/mat/files'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取所有 IMA 文件并合并为三维图像
images = []
for filename in sorted(os.listdir(ima_folder)):
    if filename.endswith('.IMA'):
        file_path = os.path.join(ima_folder, filename)
        dicom_image = pydicom.dcmread(file_path).pixel_array
        images.append(dicom_image)

# 合并为三维图像
full_volume = np.stack(images, axis=-1)  # 假设 DICOM 图像尺寸为 440x440

# 截取中心区域（128x128x128）
center_x, center_y, center_z = full_volume.shape[0] // 2, full_volume.shape[1] // 2, full_volume.shape[2] // 2
half_size = 64  # 128/2

# 截取三个方向的图像
transverse = full_volume[center_x-half_size:center_x+half_size,
                         center_y-half_size:center_y+half_size, :]
sagittal = full_volume[center_x-half_size:center_x+half_size, :, center_z-half_size:center_z+half_size]
coronal = full_volume[:, center_y-half_size:center_y+half_size, center_z-half_size:center_z+half_size]

# 保存为三个不同的 .mat 文件
scipy.io.savemat(os.path.join(output_folder, 'transverse.mat'), {'img': transverse})
scipy.io.savemat(os.path.join(output_folder, 'sagittal.mat'), {'img': sagittal})
scipy.io.savemat(os.path.join(output_folder, 'coronal.mat'), {'img': coronal})

print("处理完成，文件已保存！")
