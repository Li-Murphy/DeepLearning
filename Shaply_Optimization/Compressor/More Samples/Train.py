import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch.optim as optim
import glob
import pandas as pd

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
    return sdf/100.

sample1_data_files = glob.glob("CNN_Picture/split1/*.png")
sample2_data_files = glob.glob("CNN_Picture/split_B2_zz/*.png")
sample3_data_files = glob.glob("CNN_Picture/split_B2_zz2/*.png")
sample4_data_files = glob.glob("CNN_Picture/split_B2_zz3/*.png")


sample = [sample1_data_files,sample2_data_files,sample3_data_files,sample4_data_files]

sample1 = np.zeros([len(sample1_data_files),1108,1488])
sample2 = np.zeros([len(sample2_data_files),1108,1488])
sample3 = np.zeros([len(sample3_data_files),1108,1488])
sample4 = np.zeros([len(sample4_data_files),1108,1488])

for i in range(len(sample)):
    for j in range(len(sample[i])):
        if   i == 0:
            sample1[j] = generate_sdf(sample[i][j])
        elif i == 1:
            sample2[j] = generate_sdf(sample[i][j])
        elif i == 2:
            sample3[j] = generate_sdf(sample[i][j])
        elif i == 3:
            sample4[j] = generate_sdf(sample[i][j])
orignal = generate_sdf('CNN_Picture/orignal/Foil1.png').reshape([1,1108,1488])
all_data = np.vstack((orignal,sample1,sample2,sample3,sample4))
all_label = pd.read_excel('0.6ma_60_1.2_O.xlsx',header=None)
# 将 DataFrame 转换为 NumPy 数组
all_label = all_label.to_numpy()

np.random.seed(1)
shuffle_indices = np.random.permutation(39)
# 使用相同的打乱索引来打乱两个数组
all_data = all_data[shuffle_indices]
all_label = all_label[shuffle_indices]

train_data = torch.Tensor(all_data[:26].reshape([26,1,1108,1488])).to(device)
train_label = torch.Tensor(all_label[:26]).to(device)
validation_data = torch.Tensor(all_data[26:32].reshape([6,1,1108,1488])).to(device)
validation_label = torch.Tensor(all_label[26:32]).to(device)
test_data = torch.Tensor(all_data[32:].reshape([7,1,1108,1488])).to(device)
test_label = torch.Tensor(all_label[32:]).to(device)


all_data = torch.Tensor(all_data.reshape([39,1,1108,1488])).to(device)
all_label = torch.Tensor(all_label.reshape([39,1])).to(device)
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
        self.fc = nn.Linear(256, 125)  # 输出层
        self.relu6 = nn.ReLU()
        self.fc1 = nn.Linear(125,1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        x = self.relu6(x)
        x = self.fc1(x)
        return x


#设置损失函数，优化器
criterion = nn.MSELoss()
# 设置训练的迭代次数
epochs = 10000
# 实例化模型
model = MyCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.05)
max = 99
for epoch in range(epochs):
        # 前向传播
        outputs = model(all_data)
        # 计算损失
        loss = criterion(outputs,all_label)
        # 清空之前的梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        error = torch.mean(torch.abs((model(validation_data)-validation_label)/validation_label))
        # 打印训练状态
        if (epoch + 1) % 100 == 0:  # 每1000个epoch打印一次
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}, Error: {error}')
        if max > error:
            max = error
            torch.save(model, 'NN_All.pth')
# 训练完成
print('Training finished.')
model = torch.load('NN_All.pth')
Test_MRE = torch.mean(torch.abs(model(test_data)-test_label)/test_label)
Validation_MRE = torch.mean(torch.abs(model(validation_data)-validation_label)/validation_label)
All_MRE = torch.mean(torch.abs(model(all_data)-all_label)/all_label)
Test_MAE = torch.mean(torch.abs(model(test_data)-test_label))
Validation_MAE = torch.mean(torch.abs(model(validation_data)-validation_label))
All_MAE = torch.mean(torch.abs(model(all_data)-all_label))
print(f'Test MRE = {Test_MRE*100} % , Test MAE = {Test_MAE}')
print(f'Validation MRE = {Validation_MRE*100} % , Validation MAE = {Validation_MAE}')
print(f'ALL MRE = {All_MRE*100} % , ALL MAE = {All_MAE}')

