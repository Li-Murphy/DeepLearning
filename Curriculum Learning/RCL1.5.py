import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from CH import CHDeepONet
#导入数据
deeponet1 = torch.load('deeponet.pth')
deeponet1.eval()
RCL = torch.load('RCL0.pth')
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

train_data = pd.read_csv('data/train_0_up.csv')
test_data  = pd.read_csv('data/test_0_up.csv')
## 数据前处理 ##
all_features = pd.concat((train_data.iloc[:,2:], test_data.iloc[:,2:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features
num = train_data.shape[0]
train_x    = train_data.iloc[:num//27,0].values
train_y    = train_data.iloc[:,1].values
train_cp   = all_features.iloc[:num, 0].values
train_dun   = all_features.iloc[:num, 1].values
m    = train_x.shape[0]
x    = train_x.reshape(m,1)
y    = train_y.reshape(-1,m)
cp   = train_cp.reshape(-1,m)
dun  = train_dun.reshape(-1,m)
#设置训练集
input_u1 = torch.tensor(cp,dtype=torch.float32)
input_u2 = torch.tensor(dun,dtype=torch.float32)
input_x = torch.tensor(x,dtype=torch.float32)
output_y = torch.tensor(y,dtype=torch.float32)

#设置串形结构
branch_config = (167,[80],80)
trunk_config = (2,[80],80)
model = CHDeepONet(branch_config, trunk_config)
#设置损失函数，优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 设置训练的迭代次数
epochs = 10000
# 用于记录损失的列表
losses = []
max = 999
for epoch in range(epochs):
    a= 0
    for idx in range(input_u1.shape[0]):
        with torch.no_grad():
            net = deeponet1(input_u1[idx].unsqueeze(0), input_x)
            net1 = RCL(input_u2[idx].unsqueeze(0), net,  input_x)
            net2 = RCL1_5(input_u1[idx].unsqueeze(0), net1,  input_x)
            net3 = RCL2(input_u2[idx].unsqueeze(0),net2, input_x)
            net4 = RCL2_5(input_u1[idx].unsqueeze(0),net3,input_x)
            net5 = RCL3(input_u2[idx].unsqueeze(0), net4, input_x)
            net6 = RCL3_5(input_u1[idx].unsqueeze(0), net5, input_x)
            net7 = RCL4(input_u2[idx].unsqueeze(0), net6, input_x)
            net8 = RCL4_5(input_u1[idx].unsqueeze(0), net7, input_x)
            net9 = RCL5(input_u2[idx].unsqueeze(0), net8, input_x)
            net10 = RCL5_5(input_u1[idx].unsqueeze(0), net9, input_x)
            net11 =RCL6(input_u2[idx].unsqueeze(0), net10, input_x)
            net12 = RCL6_5(input_u1[idx].unsqueeze(0), net11, input_x)
            net13 = RCL7(input_u2[idx].unsqueeze(0), net12, input_x)
            net_1 =RCL7_5(input_u1[idx].unsqueeze(0), net13, input_x)

        outputs = model(input_u2[idx].unsqueeze(0), net_1,  input_x)
        loss = criterion(outputs, output_y[idx].unsqueeze(0))
        a = a + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if (epoch + 1) % 100 == 0:  # 每100个epoch打印一次
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
    if max > a:
        torch.save(model, "RCL0_8.pth")
        max = a
# 训练完成
print('Training finished.')
