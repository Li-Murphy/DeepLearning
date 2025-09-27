import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from DeepONet_Pytorch import DeepONet

# 定义运行硬件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 正则化代码
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

# 读取标签数据
df1 = pd.read_csv('Adjust_PLCR.csv')
PLCR = df1.to_numpy()
df1 = pd.read_csv('Adjust_PLCS.csv')
PLCS = df1.to_numpy()
df1 = pd.read_csv('Adjust_Omega.csv')
Omega = df1.to_numpy()
# 读取输入数据
df = pd.read_excel('sample_data.xlsx', sheet_name=0, header=None)
data = df.to_numpy()
data = normalize_data(data)

# 设置随机种子以确保结果的可重复性
np.random.seed(42)
# 生成一个打乱顺序的索引数组,转换为Tensor形式
shuffle_indices = np.random.permutation(data.shape[0])
data, PLCR, PLCS, Omega = torch.Tensor(data[shuffle_indices]), torch.Tensor(PLCR[shuffle_indices]), torch.Tensor(PLCS[shuffle_indices]), torch.Tensor(Omega[shuffle_indices])

# 划分数据集，训练与测试
train_size = int(data.shape[0]*0.8)
train_data, train_PLCR, train_PLCS, train_Omega = data[:train_size,:], PLCR[:train_size,:] ,PLCS[:train_size,:] ,Omega[:train_size,:]
test_data, test_PLCR, test_PLCS, test_Omega  = data[train_size:,:], PLCR[train_size:,:] ,PLCS[train_size:,:] ,Omega[train_size:,:]
# 定义Trunk层输入，并转换格式
x = np.linspace(0, 1, 100).reshape([-1,1])
x = torch.Tensor(x)

model = torch.load('Omega_0.04487353_5_1200.pth')
pred = model(test_data, x)
mae = torch.mean(torch.abs((pred - test_Omega)))

pred, x, test_PLCR, test_PLCS, test_Omega = pred.to('cpu').detach().numpy(), x.to('cpu').detach().numpy(), test_PLCR.to('cpu').detach().numpy(), test_PLCS.to('cpu').detach().numpy(), test_Omega.to('cpu').detach().numpy()
for i in range(300):
    plt.plot(pred[i,:],x,color='blue',label='pred')
    plt.plot(test_Omega[i,:],x,color='red',label='true')
    plt.title('Omega')
    plt.xlim(-0.2,1)
    plt.legend()
    plt.show()