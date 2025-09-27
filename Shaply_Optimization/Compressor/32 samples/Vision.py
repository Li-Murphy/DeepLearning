import cv2
import numpy as np
import glob
from PIL import Image
import shap
import re

sample1_data_files = glob.glob("CNN_Picture/split1/*.png")
sample2_data_files = glob.glob("CNN_Picture/split_B2_zz/*.png")
sample3_data_files = glob.glob("CNN_Picture/split_B2_zz2/*.png")
sample4_data_files = glob.glob("CNN_Picture/split_B2_zz3/*.png")

sample = [sample1_data_files,sample2_data_files,sample3_data_files,sample4_data_files]

orignal = np.array(Image.open('CNN_Picture/orignal/Foil1.png')).reshape([1,1108,1488])

sample1 = np.zeros([len(sample1_data_files),1108,1488])
sample2 = np.zeros([len(sample2_data_files),1108,1488])
sample3 = np.zeros([len(sample3_data_files),1108,1488])
sample4 = np.zeros([len(sample4_data_files),1108,1488])

for i in range(len(sample)):
    for j in range(len(sample[i])):
        if   i == 0:
            sample1[j] = np.array(Image.open(sample[i][j]))
        elif i == 1:
            sample2[j] = np.array(Image.open(sample[i][j]))
        elif i == 2:
            sample3[j] = np.array(Image.open(sample[i][j]))
        elif i == 3:
            sample4[j] = np.array(Image.open(sample[i][j]))

all_picture = np.vstack((sample1,sample3,sample4))

def sort_key(filename):
    """从文件名中提取数字，用于排序"""
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # 假设文件名中最后的数字是排序依据
    return 0
shap_data_files = glob.glob("SHAP_All/*.csv")
# 使用自定义的排序键来排序文件名列表
shap_data_files_sorted = sorted(shap_data_files, key=sort_key)

shap_values = np.zeros([31,1108,1488])

for i in range(len(shap_data_files_sorted)):
    shap_values[i] = np.loadtxt(shap_data_files[i], delimiter=",")

for i in range(len(shap_data_files_sorted)):
    shap.image_plot(shap_values[i].reshape([1,1108,1488,1]) ,all_picture[i].reshape([1,1108,1488,1]))


