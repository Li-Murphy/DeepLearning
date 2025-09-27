import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import distance_transform_edt
from PIL import Image

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

sample  = generate_sdf('Data Picture/orginal.png')
sample1 = generate_sdf('Data Picture/plot.png')
sample2 = generate_sdf('Data Picture/plot1.png')
sample3 = generate_sdf('Data Picture/plot2.png')
sample4 = generate_sdf('Data Picture/plot3.png')
sample5 = generate_sdf('Data Picture/plot4.png')
sample6 = generate_sdf('Data Picture/plot5.png')
sample7 = generate_sdf('Data Picture/plot6.png')
sample8 = generate_sdf('Data Picture/plot7.png')

Sample = np.array([sample1-sample,sample2-sample,sample3-sample,sample4-sample,sample5-sample,sample6-sample,sample7-sample,sample8-sample])

def plot_sdf(sdf_matrix):
    # 设置色彩标准化
    norm = TwoSlopeNorm(vmin=np.min(sdf_matrix), vcenter=0, vmax=np.max(sdf_matrix))
    # 创建图和轴
    plt.figure(figsize=(6, 6))
    plt.imshow(sdf_matrix, cmap='RdBu', interpolation='nearest')
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

plot_sdf(Sample[0])