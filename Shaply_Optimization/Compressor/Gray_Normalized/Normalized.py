import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


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

    return sdf


def normalize_sdf(sdf):
    # 将 SDF 数据标准化到 0-255 范围
    sdf_normalized = sdf - np.min(sdf)  # 使最小值为0
    sdf_normalized = sdf_normalized / np.max(sdf_normalized) * 255  # 归一化到 0-255
    return sdf_normalized.astype(np.uint8)  # 转换为无符号整型


def save_sdf_image(sdf, output_path):
    img = Image.fromarray(sdf)
    img.save(output_path)

sample1 = generate_sdf('plot.png')
sample2 = generate_sdf('plot1.png')
sample3 = generate_sdf('plot2.png')
sample4 = generate_sdf('plot3.png')
sample5 = generate_sdf('plot4.png')
sample6 = generate_sdf('plot5.png')
sample7 = generate_sdf('plot6.png')
sample8 = generate_sdf('plot7.png')

sample1 = normalize_sdf(sample1)
sample2 = normalize_sdf(sample2)
sample3 = normalize_sdf(sample3)
sample4 = normalize_sdf(sample4)
sample5 = normalize_sdf(sample5)
sample6 = normalize_sdf(sample6)
sample7 = normalize_sdf(sample7)
sample8 = normalize_sdf(sample8)

save_sdf_image(sample1,'sample1.png')
save_sdf_image(sample2,'sample2.png')
save_sdf_image(sample3,'sample3.png')
save_sdf_image(sample4,'sample4.png')
save_sdf_image(sample5,'sample5.png')
save_sdf_image(sample6,'sample6.png')
save_sdf_image(sample7,'sample7.png')
save_sdf_image(sample8,'sample8.png')