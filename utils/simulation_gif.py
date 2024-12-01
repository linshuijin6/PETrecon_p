import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# 假设你已经加载了脑的2D影像图像 'brain_image' 和目标弦图数据 'target_sinogram'
# brain_image 是 2D PET影像矩阵，target_sinogram 是包含模拟检测数据的弦图矩阵
brain_image = np.load("/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_picHD.npy")[0, 0, :, :]  # 载入你的脑2D PET影像
target_sinogram = np.load("/mnt/data/linshuijin/PETrecon/simulation_angular/angular_180/transverse_sinoHD.npy")[0, :, :]  # 载入你的目标弦图数据

# 设置动图的基本参数
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 初始化图像
brain_display = ax1.imshow(brain_image, cmap="gray")
sinogram_display = ax2.imshow(np.zeros_like(target_sinogram), cmap="hot", vmin=0, vmax=target_sinogram.max())
ax1.set_title("脑图 - PET模拟")
ax2.set_title("弦图 - 探测数据填充")

# 初始化状态
sinogram_data = np.zeros_like(target_sinogram)  # 空的弦图，用于动态填充

# 更新函数，用于在动画中逐帧更新
def update(frame):
    # 随机选择脑图上的发光点位置
    non_zero_indices = np.argwhere(brain_image > 0)  # 找到非零像素作为潜在发光位置
    chosen_index = random.choice(non_zero_indices)  # 随机选择一个发光位置
    brain_display.set_data(brain_image)

    # 更新发光点
    ax1.plot(chosen_index[1], chosen_index[0], 'ro')  # 用红色小圆点表示发光位置

    # 模拟检测并更新弦图数据
    angle = random.randint(0, sinogram_data.shape[1] - 1)  # 随机角度选择，模拟检测
    intensity = target_sinogram[frame % target_sinogram.shape[0], angle]  # 从目标弦图中获取相应强度
    sinogram_data[frame % target_sinogram.shape[0], angle] += intensity  # 累加强度

    # 更新弦图显示
    sinogram_display.set_data(sinogram_data)

    return brain_display, sinogram_display

# 使用FuncAnimation生成动画
ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)

# 保存动图
ani.save("pet_simulation.gif", writer="imagemagick", fps=10)
plt.show()
