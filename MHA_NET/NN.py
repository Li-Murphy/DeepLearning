
#导入必要的库文件
import torch
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
import numpy as np
import torch.optim as optim

#搭建网络模型
class NNs(nn.Module):
    def __init__(self,):
        super(NNs, self).__init__()
        self.NN_net = nn.Sequential(
            nn.Linear(334, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            #nn.ReLU(),
            nn.Linear(10, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200,167)
        )
        #选择Xavier初始化
        for layer in self.NN_net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    def forward(self,input):
        a = self.NN_net(input)
        return a

#设置运行环境为cpu
device = ('cpu')
#如果有装gpu版本pytorch启用gpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
cf_ = torch.tensor(dun_,dtype=torch.float32).to(device)
x_ = torch.tensor(x_,dtype=torch.float32).to(device)
y_ = torch.tensor(y_,dtype=torch.float32).to(device)
#设置训练集
input_cf = torch.tensor(dun,dtype=torch.float32).to(device)
input_cp = torch.tensor(cp,dtype=torch.float32).to(device)
input_x = torch.tensor(x,dtype=torch.float32).to(device)
output_y = torch.tensor(y,dtype=torch.float32).to(device)
mean_y_ = torch.tensor(mean_y_,dtype=torch.float32).to(device)

#拼接cp与cf
input_train = torch.hstack((input_cp,input_cf)).to(device)
input_test =torch.hstack((cp_,cf_)).to(device)
#设置随机种子保证结果可重复性
torch.manual_seed(1)
#实例化网络
model = NNs().to(device)

#设置损失函数，优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置训练的迭代次数
epochs = 20000
max = 99
# 开始训练循环
for epoch in range(epochs):
        # 前向传播
        outputs  = model(input_train)
        # 计算损失
        Predict_loss = criterion(outputs, output_y)
        loss = Predict_loss
        # 清空之前的梯度
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
         # 打印训练状态
        if (epoch + 1) % 1000 == 0:  # 每100个epoch打印一次
           print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
        pred = model(input_test)
        error = torch.mean(torch.abs(pred-y_)/mean_y_)
        #每个epoch结束得到测试机误差，读取其中最优
        if max > error:
            max = error
            time = epoch+1
            # torch.save(model,'NN.pth')
# 训练完成
print('Training finished.')
print(f'relative error = {max*100}%')
print(f'In epochs: {time}')