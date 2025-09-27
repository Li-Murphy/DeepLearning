import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt
import os
os.environ['OMP_NUM_THREADS'] = '1'
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
    return img_array

def normalize_sdf(sdf):
    # 将 SDF 数据标准化到 0-255 范围
    sdf_normalized = sdf - np.min(sdf)  # 使最小值为0
    sdf_normalized = sdf_normalized / np.max(sdf_normalized) * 255  # 归一化到 0-255
    return sdf_normalized.astype(np.uint8)  # 转换为无符号整型

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
sample  = generate_sdf('Data Picture/orginal.png')
sample1 = generate_sdf('Data Picture/plot.png')
sample2 = generate_sdf('Data Picture/plot1.png')
sample3 = generate_sdf('Data Picture/plot2.png')
sample4 = generate_sdf('Data Picture/plot3.png')
sample5 = generate_sdf('Data Picture/plot4.png')
sample6 = generate_sdf('Data Picture/plot5.png')
sample7 = generate_sdf('Data Picture/plot6.png')
sample8 = generate_sdf('Data Picture/plot7.png')

# sample  = normalize_sdf(sample)
# sample1 = normalize_sdf(sample1)
# sample2 = normalize_sdf(sample2)
# sample3 = normalize_sdf(sample3)
# sample4 = normalize_sdf(sample4)
# sample5 = normalize_sdf(sample5)
# sample6 = normalize_sdf(sample6)
# sample7 = normalize_sdf(sample7)
# sample8 = normalize_sdf(sample8)


train_data = np.array([sample,sample1,sample2,sample3,sample4,sample5,sample6,sample7,sample8])
train_data = train_data.reshape(9,1,1108,1488)
train_label = np.array([0.13253,0.218548,0.023273106,0.286535797,0.240166687,0.095982011,0.118230208,0.126831853,0.122568033])
train_label = train_label.reshape(9,1)

train_label = torch.Tensor(train_label)
train_data = torch.Tensor(train_data)

model = torch.load('NN_Base.pth').to('cpu')
model.eval()

explainer = shap.GradientExplainer(model, train_data[0].unsqueeze(0))
#print(train_data[1:3].shape)
shap_values = explainer.shap_values(train_data[1:])
shap_values = shap_values.reshape([8,1108,1488,1])

# a = train_data[0].unsqueeze(0)
# a = a.detach().numpy()
# a = a.reshape([1,1108,1488,1])
#shap.image_plot(shap_values, -img_array)
def plot_sdf(sdf_matrix):
    # 设置色彩标准化
    norm = TwoSlopeNorm(vmin=np.min(sdf_matrix), vcenter=0, vmax=np.max(sdf_matrix))
    # 创建图和轴
    plt.figure(figsize=(6, 6))
    plt.imshow(sdf_matrix, cmap='RdBu', interpolation='nearest',norm=norm)
    plt.colorbar()
    plt.title('Signed Distance Field')
    # 隐藏坐标轴
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # 去除图框边缘
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # 移除图例和标题
    plt.title('')
    #plt.savefig('sdf.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
    plt.show()
# shap_values =shap_values.reshape([1108,1488])
# plot_sdf(shap_values)

for i in range(8):
    a = shap_values[i]
    a = a.reshape([1108,1488])
    df = pd.DataFrame(a)
    df.to_csv(f'SHAP/plot{i+1}_Base.csv', index=False, header=False)


