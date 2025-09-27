import torch
import torch.nn as nn

# 定义Branch网络
class BranchNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(BranchNet, self).__init__()
        # 构建一个简单的全连接网络
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1], output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        return self.layers(x)


# 定义Trunk网络
class TrunkNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(TrunkNet, self).__init__()
        # 构建一个简单的全连接网络
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1], output_size)]
        layers += [nn.ReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        return self.layers(x)

# 定义DeepONet
class CHDeepONet(nn.Module):
    def __init__(self, branch_config , trunk_config):
        super(CHDeepONet, self).__init__()
        # 初始化Branch和Trunk网络
        self.branch_net = BranchNet(*branch_config)
        self.trunk_net = TrunkNet(*trunk_config)
        #添加一个偏置
        self.bias = nn.Parameter(torch.zeros(167))

    def forward(self, branch_input,net_1, trunk_input):
        # 分别对输入进行前向传播
        branch_output = self.branch_net(branch_input)
        trunk2_input = torch.cat((trunk_input, net_1.t()), dim=1)
        trunk2_output = self.trunk_net(trunk2_input)
        # 执行外积操作
        # 注意：这里假设branch_output和trunk_output的最后一个维度大小相同
        # 并且外积的结果需要求和得到最终的输出
        output = branch_output @ trunk2_output.t()+self.bias
        return output  # 对最后一个维度求和

#branch_config = (167,[80],80)
#trunk_config = (2,[80],80)
#model = DeepONet(branch_config, trunk_config)
#print(model)
