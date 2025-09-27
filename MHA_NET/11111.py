import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import numpy as np

# 生成四维数据点
x = np.array([1, 2, 3, 4])
y = np.random.standard_normal(4)
z = np.random.standard_normal(4)
w = np.random.standard_normal(4)

# 组合x, y, z到一个特征矩阵中
X = np.column_stack((x, y, z))

# 创建并拟合线性回归模型
model = LinearRegression()
model.fit(X, w)

# 创建一个网格来绘制平面
xx, yy = np.meshgrid(range(5), range(5))
zz = (model.intercept_ + model.coef_[0] * xx + model.coef_[1] * yy) / -model.coef_[2]

# 创建图表
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
ax.scatter(x, y, z, c='r', marker='o')

# 绘制拟合平面
ax.plot_surface(xx, yy, zz, alpha=0.5, rstride=1, cstride=1, color='b', shade=False)

# 设置轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图表
plt.show()
