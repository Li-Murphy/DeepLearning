import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
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
    # min = np.min(sdf)
    # sdf = sdf - min
    return sdf/100.

def normalize_sdf(sdf):
    # 将 SDF 数据标准化到 0-255 范围
    sdf_normalized = (sdf - np.mean(sdf)) / np.std(sdf)  # 使最小值为0
    return sdf_normalized # 转换为无符号整型

sample1 = generate_sdf('Data Picture/plot.png')
sample2 = generate_sdf('Data Picture/plot1.png')
sample3 = generate_sdf('Data Picture/plot2.png')
sample4 = generate_sdf('Data Picture/plot3.png')
sample5 = generate_sdf('Data Picture/plot4.png')
sample6 = generate_sdf('Data Picture/plot5.png')
sample7 = generate_sdf('Data Picture/plot6.png')
sample8 = generate_sdf('Data Picture/plot7.png')
sample  = generate_sdf('Data Picture/orginal.png')

# sample1 = normalize_sdf(sample1)
# sample2 = normalize_sdf(sample2)
# sample3 = normalize_sdf(sample3)
# sample4 = normalize_sdf(sample4)
# sample5 = normalize_sdf(sample5)
# sample6 = normalize_sdf(sample6)
# sample7 = normalize_sdf(sample7)
# sample8 = normalize_sdf(sample8)
# sample = normalize_sdf(sample)

train_data = np.array([sample,sample1,sample2,sample3,sample4,sample5,sample6,sample7,sample8])
train_data = train_data.reshape(9,1,1108,1488)
train_label = np.array([0.145336524747737,0.0570865794138368,0.0462602438717726,0.20893125483946,0.154697231859122,0.0981115677608138,0.107410140542335,0.0772596916029477,0.117699751486743])
train_label = train_label.reshape(9,1)
# 定义网络结构
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # 设计卷积层以逐步减小特征图尺寸至1x1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=5)  # Output: 16 x 277 x 372
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=11, stride=4, padding=5)  # Output: 32 x 70 x 94
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=11, stride=4, padding=5)  # Output: 64 x 18 x 24
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 128, kernel_size=7, stride=3, padding=3)  # Output: 128 x 6 x 8
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 256, kernel_size=6, stride=6, padding=0)  # Output: 256 x 1 x 1
        self.relu5 = nn.ReLU()

        # 最终的卷积层输出尺寸为1x1x256，直接连接到输出层
        self.fc = nn.Linear(256, 1)  # 输出层

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


#设置损失函数，优化器
criterion = nn.MSELoss()
# 设置训练的迭代次数
epochs = 10000
# 实例化模型
model = MyCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00005)
train_label = torch.Tensor(train_label).to(device)
train_data = torch.Tensor(train_data).to(device)
max = 99
for epoch in range(epochs):
        # 前向传播
        outputs = model(train_data)
        # 计算损失
        loss = criterion(outputs,train_label)
        # 清空之前的梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        error = torch.mean(torch.abs(outputs-train_label)/train_label)
        # 打印训练状态
        if (epoch + 1) % 100 == 0:  # 每1000个epoch打印一次
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Error: {error}')
        if max > error:
            max = error
            a = outputs
            torch.save(model, 'NN_Base.pth')
# 训练完成
print('Training finished.')
print(a)
print(train_label)

