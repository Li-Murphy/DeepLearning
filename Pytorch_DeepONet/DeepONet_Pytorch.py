import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, branchinput_dim,trunkinput_dim, layers, hidden_dim):
        super(DeepONet, self).__init__()

        # Branch network
        branch_modules = [nn.Linear(branchinput_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 2):
            branch_modules.append(nn.Linear(hidden_dim, hidden_dim))
            branch_modules.append(nn.ReLU())
        branch_modules.append(nn.Linear(hidden_dim, hidden_dim))  # Last layer without ReLU
        self.branch_net = nn.Sequential(*branch_modules)

        # Trunk network
        trunk_modules = [nn.Linear(trunkinput_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 1):
            trunk_modules.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_modules.append(nn.ReLU())
        self.trunk_net = nn.Sequential(*trunk_modules)


    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Element-wise multiplication and sum
        output = branch_output @ trunk_output.t()

        return output