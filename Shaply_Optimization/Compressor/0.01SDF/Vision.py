import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

s1 = np.loadtxt('SHAP/plot1_Base.csv', delimiter=",")
s2 = np.loadtxt('SHAP/plot2_Base.csv', delimiter=",")
s3 = np.loadtxt('SHAP/plot3_Base.csv', delimiter=",")
s4 = np.loadtxt('SHAP/plot4_Base.csv', delimiter=",")
s5 = np.loadtxt('SHAP/plot5_Base.csv', delimiter=",")
s6 = np.loadtxt('SHAP/plot6_Base.csv', delimiter=",")
s7 = np.loadtxt('SHAP/plot7_Base.csv', delimiter=",")
s8 = np.loadtxt('SHAP/plot8_Base.csv', delimiter=",")
shap_values = np.array([s1,s2,s3,s4,s4,s6,s7,s8])
print(np.sum(s1))

img1 = Image.open('Data Picture/plot.png')
img2 = Image.open('Data Picture/plot1.png')
img3 = Image.open('Data Picture/plot2.png')
img4 = Image.open('Data Picture/plot3.png')
img5 = Image.open('Data Picture/plot4.png')
img6 = Image.open('Data Picture/plot5.png')
img7 = Image.open('Data Picture/plot6.png')
img8 = Image.open('Data Picture/plot7.png')
# 将图像转换为numpy数组
img1 = np.array(img1)
img2 = np.array(img2)
img3 = np.array(img3)
img4 = np.array(img4)
img5 = np.array(img5)
img6 = np.array(img6)
img7 = np.array(img7)
img8 = np.array(img8)
img_array = np.array([img1,img2,img3,img4,img5,img6,img7,img8])
img_array = img_array.reshape([8,1108,1488])

# for i in range(8):
#     # 设置色彩标准化
#     norm = TwoSlopeNorm(vmin=np.min(shap_values[i]), vcenter=0, vmax=np.max(shap_values[i]))
#     # 创建图和轴
#     plt.figure(figsize=(6, 6))
#     #plt.imshow(img_array[i],cmap='gray', alpha=0.5)
#     plt.imshow(shap_values[i], cmap='RdBu', interpolation='nearest', norm=norm)
#
#     plt.colorbar()
#     #plt.savefig(f'plot{i+1}_Base.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0)
#     plt.show()


for i in range(8):
    shap.image_plot(shap_values[i].reshape([1,1108,1488,1])*10 ,-img_array[i].reshape([1,1108,1488,1]))
