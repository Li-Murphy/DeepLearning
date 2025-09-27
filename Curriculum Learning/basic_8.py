import torch
import torch.nn as nn
from Curriculum_Learning import DeepONet
import pandas as pd
import torch.optim as optim
#构建模型
branch_config = (167,[80],80)
trunk1_config = (1,[80],80)
trunk2_config = (2,[80],80)
model = DeepONet(branch_config, trunk1_config,trunk2_config)

#导入数据
train_data = pd.read_csv('data/train_8_up.csv')
test_data  = pd.read_csv('data/test_8_up.csv')
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
#设置损失函数，优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 设置训练的迭代次数
epochs = 10000
# 用于记录损失的列表
losses = []
max = 999
for epoch in range(epochs):
    a= 0
    for idx in range(input_u1.shape[0]):
        outputs = model(input_u1[idx].unsqueeze(0), input_u2[idx].unsqueeze(0),  input_x)
        loss = criterion(outputs, output_y[idx].unsqueeze(0))
        a = a + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if (epoch + 1) % 100 == 0:  # 每100个epoch打印一次
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
    if max > a:
        torch.save(model, "CL8.pth")
        max = a
# 训练完成
print('Training finished.')
