import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import TwoSlopeNorm

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
all_data = np.vstack((orignal,sample1,sample3,sample4))

all_data = torch.Tensor(all_data.reshape([32,1,1108,1488]))


model = torch.load('NN_All_P.pth').to('cpu')
model.eval()

explainer = shap.GradientExplainer(model, all_data[0].unsqueeze(0))
#print(train_data[1:3].shape)
shap_values = explainer.shap_values(all_data[1:])
shap_values = shap_values.reshape([31,1108,1488,1])

for i in range(31):
    a = shap_values[i]
    a = a.reshape([1108,1488])
    df = pd.DataFrame(a)
    df.to_csv(f'SHAP_All/plot_{i+1}_Base.csv', index=False, header=False)