import cv2
import numpy as np

# 读取图片
image = cv2.imread('orginal.png', cv2.IMREAD_GRAYSCALE)

# 应用阈值处理将图片转换为二值图像
_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# 寻找所有轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个全黑的图像
filled_image = np.zeros_like(binary)

# 填充所有轮廓
for contour in contours:
    cv2.drawContours(filled_image, [contour], -1, (255), thickness=cv2.FILLED)

# 显示结果
cv2.imshow('Filled Image', filled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
#cv2.imwrite('orginal.png', filled_image)

