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
class DeepONet(nn.Module):
    def __init__(self, branch_config, trunk1_config, trunk2_config):
        super(DeepONet, self).__init__()
        # 初始化Branch和Trunk网络
        self.branch1_net = BranchNet(*branch_config)
        self.branch2_net = BranchNet(*branch_config)
        self.branch3_net = BranchNet(*branch_config)
        self.branch4_net = BranchNet(*branch_config)
        self.branch5_net = BranchNet(*branch_config)
        self.branch6_net = BranchNet(*branch_config)
        self.branch7_net = BranchNet(*branch_config)
        self.branch8_net = BranchNet(*branch_config)
        self.branch9_net = BranchNet(*branch_config)
        self.branch10_net = BranchNet(*branch_config)
        self.branch11_net = BranchNet(*branch_config)
        self.branch12_net = BranchNet(*branch_config)
        self.branch13_net = BranchNet(*branch_config)
        self.branch14_net = BranchNet(*branch_config)
        self.branch15_net = BranchNet(*branch_config)
        self.branch16_net = BranchNet(*branch_config)
        self.branch17_net = BranchNet(*branch_config)
        self.branch18_net = BranchNet(*branch_config)
        self.branch19_net = BranchNet(*branch_config)
        self.branch20_net = BranchNet(*branch_config)

        self.trunk1_net = TrunkNet(*trunk1_config)
        self.trunk2_net = TrunkNet(*trunk2_config)
        self.trunk3_net = TrunkNet(*trunk2_config)
        self.trunk4_net = TrunkNet(*trunk2_config)
        self.trunk5_net = TrunkNet(*trunk2_config)
        self.trunk6_net = TrunkNet(*trunk2_config)
        self.trunk7_net = TrunkNet(*trunk2_config)
        self.trunk8_net = TrunkNet(*trunk2_config)
        self.trunk9_net = TrunkNet(*trunk2_config)
        self.trunk10_net = TrunkNet(*trunk2_config)
        self.trunk11_net = TrunkNet(*trunk2_config)
        self.trunk12_net = TrunkNet(*trunk2_config)
        self.trunk13_net = TrunkNet(*trunk2_config)
        self.trunk14_net = TrunkNet(*trunk2_config)
        self.trunk15_net = TrunkNet(*trunk2_config)
        self.trunk16_net = TrunkNet(*trunk2_config)
        self.trunk17_net = TrunkNet(*trunk2_config)
        self.trunk18_net = TrunkNet(*trunk2_config)
        self.trunk19_net = TrunkNet(*trunk2_config)
        self.trunk20_net = TrunkNet(*trunk2_config)
        #添加一个偏置
        self.bias1 = nn.Parameter(torch.zeros(167))
        self.bias2 = nn.Parameter(torch.zeros(167))
        self.bias3 = nn.Parameter(torch.zeros(167))
        self.bias4 = nn.Parameter(torch.zeros(167))
        self.bias5 = nn.Parameter(torch.zeros(167))
        self.bias6 = nn.Parameter(torch.zeros(167))
        self.bias7 = nn.Parameter(torch.zeros(167))
        self.bias8 = nn.Parameter(torch.zeros(167))
        self.bias9 = nn.Parameter(torch.zeros(167))
        self.bias10 = nn.Parameter(torch.zeros(167))
        self.bias11 = nn.Parameter(torch.zeros(167))
        self.bias12 = nn.Parameter(torch.zeros(167))
        self.bias13 = nn.Parameter(torch.zeros(167))
        self.bias14 = nn.Parameter(torch.zeros(167))
        self.bias15 = nn.Parameter(torch.zeros(167))
        self.bias16 = nn.Parameter(torch.zeros(167))
        self.bias17 = nn.Parameter(torch.zeros(167))
        self.bias18 = nn.Parameter(torch.zeros(167))
        self.bias19 = nn.Parameter(torch.zeros(167))
        self.bias20 = nn.Parameter(torch.zeros(167))


    def forward(self, branch1_input, branch2_input, trunk_input):
        # 0.5
        branch1_output = self.branch1_net(branch1_input)
        trunk1_output = self.trunk1_net(trunk_input)
        net_1 = branch1_output @ trunk1_output.t()+self.bias1
        # 1
        trunk2_input = torch.cat((trunk_input, net_1.t()), dim=1)
        trunk2_output = self.trunk2_net(trunk2_input)
        branch2_output = self.branch2_net(branch2_input)
        net_2 = branch2_output @ trunk2_output.t() + self.bias2
        # # 1.5
        # trunk3_input = torch.cat((trunk_input, net_2.t()), dim=1)
        # trunk3_output = self.trunk3_net(trunk3_input)
        # branch3_output = self.branch3_net(branch1_input)
        # net_3 = branch3_output @ trunk3_output.t() + self.bias3
        # # 2
        # trunk4_input = torch.cat((trunk_input, net_3.t()), dim=1)
        # trunk4_output = self.trunk4_net(trunk4_input)
        # branch4_output = self.branch4_net(branch2_input)
        # net_4 = branch4_output @ trunk4_output.t() + self.bias4
        # # 2.5
        # trunk5_input = torch.cat((trunk_input, net_4.t()), dim=1)
        # trunk5_output = self.trunk5_net(trunk5_input)
        # branch5_output = self.branch5_net(branch1_input)
        # net_5 = branch5_output @ trunk5_output.t() + self.bias5
        # # 3
        # trunk6_input = torch.cat((trunk_input, net_5.t()), dim=1)
        # trunk6_output = self.trunk6_net(trunk6_input)
        # branch6_output = self.branch6_net(branch2_input)
        # net_6 = branch6_output @ trunk6_output.t() + self.bias6
        # # 3.5
        # trunk7_input = torch.cat((trunk_input, net_6.t()), dim=1)
        # trunk7_output = self.trunk7_net(trunk7_input)
        # branch7_output = self.branch7_net(branch1_input)
        # net_7 = branch7_output @ trunk7_output.t() + self.bias7
        # # 4
        # trunk8_input = torch.cat((trunk_input, net_7.t()), dim=1)
        # trunk8_output = self.trunk8_net(trunk8_input)
        # branch8_output = self.branch8_net(branch2_input)
        # net_8 = branch8_output @ trunk8_output.t() + self.bias8
        # # 4.5
        # trunk9_input = torch.cat((trunk_input, net_8.t()), dim=1)
        # trunk9_output = self.trunk9_net(trunk9_input)
        # branch9_output = self.branch9_net(branch1_input)
        # net_9 = branch9_output @ trunk9_output.t() + self.bias9
        # # 5
        # trunk10_input = torch.cat((trunk_input, net_9.t()), dim=1)
        # trunk10_output = self.trunk10_net(trunk10_input)
        # branch10_output = self.branch10_net(branch2_input)
        # net_10 = branch10_output @ trunk10_output.t() + self.bias10
        # # 5.5
        # trunk11_input = torch.cat((trunk_input, net_10.t()), dim=1)
        # trunk11_output = self.trunk11_net(trunk11_input)
        # branch11_output = self.branch11_net(branch1_input)
        # net_11 = branch11_output @ trunk11_output.t() + self.bias11
        # # 6
        # trunk12_input = torch.cat((trunk_input, net_11.t()), dim=1)
        # trunk12_output = self.trunk12_net(trunk12_input)
        # branch12_output = self.branch12_net(branch2_input)
        # net_12 = branch12_output @ trunk12_output.t() + self.bias12
        # # 6.5
        # trunk13_input = torch.cat((trunk_input, net_12.t()), dim=1)
        # trunk13_output = self.trunk13_net(trunk13_input)
        # branch13_output = self.branch13_net(branch1_input)
        # net_13 = branch13_output @ trunk13_output.t() + self.bias13
        # # 7
        # trunk14_input = torch.cat((trunk_input, net_13.t()), dim=1)
        # trunk14_output = self.trunk14_net(trunk14_input)
        # branch14_output = self.branch14_net(branch2_input)
        # net_14 = branch14_output @ trunk14_output.t() + self.bias14
        # # 7.5
        # trunk15_input = torch.cat((trunk_input, net_14.t()), dim=1)
        # trunk15_output = self.trunk15_net(trunk15_input)
        # branch15_output = self.branch15_net(branch1_input)
        # net_15 = branch15_output @ trunk15_output.t() + self.bias15
        # # 8
        # trunk16_input = torch.cat((trunk_input, net_15.t()), dim=1)
        # trunk16_output = self.trunk16_net(trunk16_input)
        # branch16_output = self.branch16_net(branch2_input)
        # net_16 = branch16_output @ trunk16_output.t() + self.bias16
        # # 8.5
        # trunk17_input = torch.cat((trunk_input, net_16.t()), dim=1)
        # trunk17_output = self.trunk17_net(trunk17_input)
        # branch17_output = self.branch17_net(branch1_input)
        # net_17 = branch17_output @ trunk17_output.t() + self.bias17
        # # 9
        # trunk18_input = torch.cat((trunk_input, net_17.t()), dim=1)
        # trunk18_output = self.trunk18_net(trunk18_input)
        # branch18_output = self.branch18_net(branch2_input)
        # net_18 = branch18_output @ trunk18_output.t() + self.bias18
        # # 9.5
        # trunk19_input = torch.cat((trunk_input, net_18.t()), dim=1)
        # trunk19_output = self.trunk19_net(trunk19_input)
        # branch19_output = self.branch19_net(branch1_input)
        # net_19 = branch19_output @ trunk19_output.t() + self.bias19
        # # 10
        # trunk20_input = torch.cat((trunk_input, net_19.t()), dim=1)
        # trunk20_output = self.trunk20_net(trunk20_input)
        # branch20_output = self.branch20_net(branch2_input)
        # output = branch20_output @ trunk20_output.t() + self.bias20
        return net_2  # 对最后一个维度求和

#branch_config = (167,[80],80)
#trunk1_config = (1,[80],80)
#trunk2_config = (2,[80],80)
#deeponet = DeepONet(branch_config, trunk1_config,trunk2_config)
#print(deeponet)
#branch1 = torch.ones(1,167)
#branch2 = torch.ones(1,167)
#trunk1 = torch.ones(167,1)
#out = deeponet(branch1,branch2,trunk1)
#print(out.shape)