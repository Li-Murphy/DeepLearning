import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

deeponet1 = torch.load('deeponet.pth')
CL = torch.load('CL0.pth')
RCL = torch.load('RCL0.pth')
RCL99 = torch.load('RCL0_99.pth')
RCL1_5 = torch.load('RCL0_1.5.pth')
RCL2 = torch.load('RCL0_2.pth')
RCL2_5 = torch.load('RCL0_2.5.pth')
RCL3 = torch.load('RCL0_3.pth')
RCL3_5 = torch.load('RCL0_3.5.pth')
RCL4 = torch.load('RCL0_4.pth')
RCL4_5 = torch.load('RCL0_4.5.pth')
RCL5 = torch.load('RCL0_5.pth')
RCL5_5 = torch.load('RCL0_5.5.pth')
RCL6 = torch.load('RCL0_6.pth')
RCL6_5 = torch.load('RCL0_6.5.pth')
RCL7 = torch.load('RCL0_7.pth')
RCL7_5 = torch.load('RCL0_7.5.pth')
RCL8 = torch.load('RCL0_8.pth')

## 数据导入
train_data = pd.read_csv('data/train_0_up.csv')
test_data = pd.read_csv('data/test_0_up.csv')

# 正则化
all_features = pd.concat((train_data.iloc[:, 2:], test_data.iloc[:, 2:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

num = train_data.shape[0]
train_x = train_data.iloc[:num // 27, 0].values
test_x = test_data.iloc[:num // 27, 0].values
test_y = test_data.iloc[:, 1].values
test_cp = all_features.iloc[num:, 0].values
test_dun = all_features.iloc[num:, 1].values

# 设置输入训练参数
m = train_x.shape[0]
x_ = test_x.reshape(m, -1)
y_ = test_y.reshape(-1, m)
cp_ = test_cp.reshape(-1, m)
dun_ = test_dun.reshape(-1, m)

# 设置测试集
u_ = torch.tensor(dun_, dtype=torch.float32)
x_ = torch.tensor(x_, dtype=torch.float32)
y_ = torch.tensor(y_, dtype=torch.float32)
cp_ = torch.tensor(cp_, dtype=torch.float32)
y1 = torch.mean(y_)

y_s = torch.zeros(150,167)
y_1 = torch.zeros(150,167)
y_1_5 = torch.zeros(150,167)
y_2 = torch.zeros(150,167)
y_2_5 = torch.zeros(150,167)
y_3 = torch.zeros(150,167)
y_3_5 = torch.zeros(150,167)
y_4 = torch.zeros(150,167)
y_4_5 = torch.zeros(150,167)
y_5 = torch.zeros(150,167)
y_5_5 = torch.zeros(150,167)
y_6 = torch.zeros(150,167)
y_6_5 = torch.zeros(150,167)
y_7 = torch.zeros(150,167)
y_7_5 = torch.zeros(150,167)
y_8 = torch.zeros(150,167)
y_99 = torch.zeros(150,167)

for idx in range(u_.shape[0]):
    y_s[idx] = CL(cp_[idx].unsqueeze(0),u_[idx].unsqueeze(0),x_)
    net_1 = deeponet1(cp_[idx].unsqueeze(0),x_)
    y_1[idx] = RCL(u_[idx].unsqueeze(0),net_1,x_)
    y_99[idx] = RCL99(cp_[idx].unsqueeze(0),y_1[idx].unsqueeze(0),x_)
    y_1_5[idx] = RCL1_5(cp_[idx].unsqueeze(0),y_1[idx].unsqueeze(0),x_)
    y_2[idx] = RCL2(u_[idx].unsqueeze(0), y_1_5[idx].unsqueeze(0), x_)
    y_2_5[idx] = RCL2_5(cp_[idx].unsqueeze(0),y_2[idx].unsqueeze(0),x_)
    y_3[idx] = RCL3(u_[idx].unsqueeze(0), y_2_5[idx].unsqueeze(0), x_)
    y_3_5[idx] =RCL3_5(cp_[idx].unsqueeze(0),y_3[idx].unsqueeze(0),x_)
    y_4[idx] = RCL4(u_[idx].unsqueeze(0), y_3_5[idx].unsqueeze(0), x_)
    y_4_5[idx] = RCL4_5(cp_[idx].unsqueeze(0), y_4[idx].unsqueeze(0), x_)
    y_5[idx] = RCL5(u_[idx].unsqueeze(0), y_4_5[idx].unsqueeze(0), x_)
    y_5_5[idx] = RCL5_5(cp_[idx].unsqueeze(0), y_5[idx].unsqueeze(0), x_)
    y_6[idx]= RCL6(u_[idx].unsqueeze(0), y_5_5[idx].unsqueeze(0), x_)
    y_6_5[idx] = RCL6_5(cp_[idx].unsqueeze(0), y_6[idx].unsqueeze(0), x_)
    y_7[idx] = RCL7(u_[idx].unsqueeze(0), y_6_5[idx].unsqueeze(0), x_)
    y_7_5[idx] = RCL7_5(cp_[idx].unsqueeze(0), y_7[idx].unsqueeze(0), x_)
    y_8[idx] = RCL8(u_[idx].unsqueeze(0), y_7_5[idx].unsqueeze(0), x_)

print(y_1.shape,y_s.shape)
mape = torch.mean(torch.abs((y_ - y_99) / y1))
mape1 = torch.mean(torch.abs((y_ - y_1) / y1))
mape1_5 = torch.mean(torch.abs((y_ - y_1_5) / y1))
mape2 = torch.mean(torch.abs((y_ - y_2) / y1))
mape2_5 = torch.mean(torch.abs((y_ - y_2_5) / y1))
mape3 = torch.mean(torch.abs((y_ - y_3) / y1))
mape3_5 = torch.mean(torch.abs((y_ - y_3_5) / y1))
mape4 = torch.mean(torch.abs((y_ - y_4) / y1))
mape4_5 = torch.mean(torch.abs((y_ - y_4_5) / y1))
mape5 = torch.mean(torch.abs((y_ - y_5) / y1))
mape5_5 = torch.mean(torch.abs((y_ - y_5_5) / y1))
mape6 = torch.mean(torch.abs((y_ - y_6) / y1))
mape6_5 = torch.mean(torch.abs((y_ - y_6_5) / y1))
mape7 = torch.mean(torch.abs((y_ - y_7) / y1))
mape7_5 = torch.mean(torch.abs((y_ - y_7_5) / y1))
mape8 = torch.mean(torch.abs((y_ - y_8) / y1))


print(f"1平均相对误差: {mape1.item() * 100:.2f}%")
print(f"1.5平均相对误差: {mape.item() * 100:.2f}%")
print(f"2平均相对误差: {mape2.item() * 100:.2f}%")
print(f"2.5平均相对误差: {mape2_5.item() * 100:.2f}%")
print(f"3平均相对误差: {mape3.item() * 100:.2f}%")
print(f"3.5平均相对误差: {mape3_5.item() * 100:.2f}%")
print(f"4平均相对误差: {mape4.item() * 100:.2f}%")
print(f"4.5平均相对误差: {mape4_5.item() * 100:.2f}%")
print(f"5平均相对误差: {mape5.item() * 100:.2f}%")
print(f"5.5平均相对误差: {mape5_5.item() * 100:.2f}%")
print(f"6平均相对误差: {mape6.item() * 100:.2f}%")
print(f"6.5平均相对误差: {mape6_5.item() * 100:.2f}%")
print(f"7平均相对误差: {mape7.item() * 100:.2f}%")
print(f"7.5平均相对误差: {mape7_5.item() * 100:.2f}%")
print(f"8平均相对误差: {mape8.item() * 100:.2f}%")
