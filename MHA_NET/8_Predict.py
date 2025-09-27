from MHA import MHA
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## 数据导入
train_data = pd.read_csv('train_8_up.csv')
test_data  = pd.read_csv('test_8_up.csv')

## 数据前处理 ##

# 正则化
all_features = pd.concat((train_data.iloc[:,2:], test_data.iloc[:,2:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

num = train_data.shape[0]
train_x    = train_data.iloc[:num//27,0].values
train_y    = train_data.iloc[:,1].values
train_cp   = all_features.iloc[:num, 0].values
train_dun   = all_features.iloc[:num, 1].values

test_x    = test_data.iloc[:num//27,0].values
test_y    = test_data.iloc[:,1].values
test_cp   = all_features.iloc[num:, 0].values
test_dun   = all_features.iloc[num:, 1].values

m    = train_x.shape[0]
x    = train_x.reshape(m,1)
y    = train_y.reshape(-1,m)
cp   = train_cp.reshape(-1,m)
dun  = train_dun.reshape(-1,m)

x_   = test_x.reshape(m,1)
y_   = test_y.reshape(-1,m)
cp_  = test_cp.reshape(-1,m)
dun_ = test_dun.reshape(-1,m)
mean_y_ = np.mean(y_,axis=1,keepdims=True)
#设置测试集
cp_ = torch.tensor(cp_,dtype=torch.float32).to(device)
u_ = torch.tensor(dun_,dtype=torch.float32).to(device)
x_ = torch.tensor(x_,dtype=torch.float32).to(device)
y_ = torch.tensor(y_,dtype=torch.float32).to(device)

#设置训练集
input_u = torch.tensor(dun,dtype=torch.float32).to(device)
input_cp = torch.tensor(cp,dtype=torch.float32).to(device)
input_x = torch.tensor(x,dtype=torch.float32).to(device)
output_y = torch.tensor(y,dtype=torch.float32).to(device)
mean_y_ = torch.tensor(mean_y_,dtype=torch.float32).to(device)

#定义网络参数
Encoder_config = (167,[100,50,25],10)
Decoder_config = (10,[25,50,100],167)
FC1_config = (167,[200,200],10)
FC2_config = (10,[200,200],167)
#Trunk_config = (1,[200,200],200)
torch.manual_seed(1)
#设置损失函数，优化器
criterion = nn.MSELoss()
# 设置训练的迭代次数
epochs = 30000
P_l = []
A_l = []
F_l = []
l = []
# 开始训练循环
max = 99
model = MHA(Encoder_config, Decoder_config, FC1_config, FC2_config).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
for epoch in range(epochs):
        # 前向传播
        outputs , Encoder , FC , Decoder = model(input_cp,input_cp)
        # 计算损失
        Predict_loss = criterion(outputs, output_y)
        AE_Loss = criterion(input_u ,Decoder)
        Fusion_Loss = criterion(FC , Encoder)
        loss = Predict_loss + AE_Loss + Fusion_Loss
        # 清空之前的梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        l.append(loss.item())
        P_l.append(Predict_loss.item())
        A_l.append(AE_Loss.item())
        F_l.append(Fusion_Loss.item())
        # 打印训练状态
        if (epoch + 1) % 10000 == 0:  # 每1000个epoch打印一次
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
        pred , E , F , D = model(cp_,u_)
        error = torch.mean(torch.abs(pred-y_)/mean_y_)
        if max > error:
                max = error
                time = epoch + 1
# 训练完成
print('Training finished.')
#print(f'weight = {0.1*i}')
print(f'relative error = {max*100}%')
print(f'In epochs: {time}')

# 绘制损失图
plt.plot(l,label='total')
plt.plot(P_l,label='Predict')
plt.plot(A_l,label='Auto_Decoder')
plt.plot(F_l,label='Fusion')
plt.title('Loss vs. Epochs')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

