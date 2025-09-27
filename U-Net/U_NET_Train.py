import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from U_NET import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_sdf(image_path):
    # 加载图像并转换为灰度
    img = Image.open(image_path).convert('L')

    # 将图像转换为二值数组
    img_array = np.array(img)
    binary_image = img_array < 128  # 假设阈值为128来区分黑白

    # 计算距离变换
    dist_transform_inside = distance_transform_edt(binary_image)  # 翼型内部到边界的距离
    dist_transform_outside = distance_transform_edt(1 - binary_image)  # 背景到翼型边界的距离

    # 创建符号距离场，内部为负，外部为正
    sdf = dist_transform_inside - dist_transform_outside

    return sdf/100.
def normalize_sdf(sdf):
    # 将 SDF 数据标准化到 0-255 范围
    sdf_normalized = (sdf - np.mean(sdf)) / np.std(sdf)  # 使最小值为0
    return sdf_normalized # 转换为无符号整型

# 自定义 MSE 损失函数，忽略 0 项
def masked_mse_loss(input, target, mask):
    diff = (input - target) ** 2
    masked_diff = diff * mask
    return masked_diff.sum() / mask.sum()
# 读取包含四位数的文件
with open('file_list.txt', 'r') as file:
    numbers = file.readlines()
# 去除每行的换行符
numbers = [num.strip() for num in numbers]
a = np.zeros([125,256,256])
b = np.zeros([125,512*512,4])
for i in range(125):
    a[i] = normalize_sdf(generate_sdf(f'Data/BW_Picture/{numbers[i]}.png'))
    b[i] = pd.read_csv(f'Data/AOA4_0.3MA_Flow_Field/{numbers[i]}.dat',sep='\s+',skiprows=19,names=['X', 'Y', 'P','T'],header=None)

data = torch.Tensor(a).reshape([125,1,256,256])
l = torch.Tensor(b)
x = l[:,:,0].reshape([125,1,512,512])
y = l[:,:,1].reshape([125,1,512,512])
pressure = l[:,:,2].reshape([125,1,512,512])
temperature = l[:,:,3].reshape([125,1,512,512])
label = torch.cat((pressure,temperature),dim=1)
torch.manual_seed(42)
indices = torch.randperm(data.size(0))
data = data[indices,:,:,:]
label= label[indices,:,:,:]
pressure = pressure[indices,:,:,:]
temperature = temperature[indices,:,:,:]
torch.manual_seed(1)
train_data , train_label = data[:25].to(device) , label[:25].to(device)
test_data , test_label = data[100:].to(device) , label[100:].to(device)
model = UNet().to(device)
#设置损失函数，优化器
# 设置训练的迭代次数
epochs = 10000
# 开始训练循环
max = 9999999999
optimizer = optim.Adam(model.parameters(), lr=0.5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, cooldown=10)
for epoch in range(epochs):
    j = torch.randperm(train_data.size(0))[:60]
    td,tl = train_data[j,:,:,:],train_label[j,:,:,:]
    output = model(td)
    output[ tl == 0. ] = 0.
    # 生成一个掩码，忽略所有 A 中值为 0 的位置
    mask = (tl != 0).float()
    loss = masked_mse_loss(output,tl,mask)
    # 清空之前的梯度
    optimizer.zero_grad()
    # 反向传播，计算梯度
    loss.backward()
    # 更新参数
    optimizer.step()
    pred = model(test_data)
    pred[ test_label==0. ] = 0.
    mask_t = (test_label != 0 )
    error = torch.abs((pred - test_label) / test_label)
    error = torch.mean(error[mask_t])
    # 调度学习率
    scheduler.step(loss)
    # 打印训练状态
    if (epoch + 1) % 10 == 0:  # 每1000个epoch打印一次
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Test MRE: {error*100}')
    if max > error:
        max = error
        t = epoch + 1
        # 保存模型的状态字典
        torch.save(model.state_dict(), 'unet_model.pth')

print('Training finished.')
print(f'relative error = {max * 100}%')
print(f'In epochs: {t}')

