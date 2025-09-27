import torch
import torch.nn as nn
from torch.nn import init

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_layers,output_size):
        super(Encoder,self).__init__()
        layers = [nn.Linear(input_size,hidden_layers[0]),nn.ReLU()]
        if len(hidden_layers) > 1:
            for i in range(1 , len(hidden_layers)):
                layers += [nn.Linear(hidden_layers[i-1],hidden_layers[i]),nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1],output_size)]
        self.layers = nn.Sequential(*layers)
        # 初始化
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self,input_size,hidden_layers,output_size):
        super(Decoder,self).__init__()
        layers = [nn.Linear(input_size,hidden_layers[0]),nn.ReLU()]
        if len(hidden_layers) > 1:
            for i in range(1 , len(hidden_layers)):
                layers += [nn.Linear(hidden_layers[i-1],hidden_layers[i]),nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1],output_size)]
        self.layers = nn.Sequential(*layers)
        #初始化
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)

class FC1(nn.Module):
    def __init__(self,input_size,hidden_layers,output_size):
        super(FC1,self).__init__()
        layers = [nn.Linear(input_size,hidden_layers[0]),nn.ReLU()]
        if len(hidden_layers) > 1:
            for i in range(1 , len(hidden_layers)):
                layers += [nn.Linear(hidden_layers[i-1],hidden_layers[i]),nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1],output_size)]
        self.layers = nn.Sequential(*layers)
        # 初始化
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)

class FC2(nn.Module):
    def __init__(self,input_size,hidden_layers,output_size):
        super(FC2,self).__init__()
        layers = [nn.Linear(input_size,hidden_layers[0]),nn.ReLU()]
        if len(hidden_layers) > 1:
            for i in range(1 , len(hidden_layers)):
                layers += [nn.Linear(hidden_layers[i-1],hidden_layers[i]),nn.ReLU()]
        layers += [nn.Linear(hidden_layers[-1],output_size)]
        self.layers = nn.Sequential(*layers)
        # 初始化
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)

# class Trunk(nn.Module):
#     def __init__(self,input_size,hidden_layers,output_size):
#         super(Trunk,self).__init__()
#         layers = [nn.Linear(input_size,hidden_layers[0]),nn.ReLU()]
#         if len(hidden_layers) > 1:
#             for i in range(1 , len(hidden_layers)):
#                 layers += [nn.Linear(hidden_layers[i-1],hidden_layers[i]),nn.ReLU()]
#         layers += [nn.Linear(hidden_layers[-1],output_size)]
#         self.layers = nn.Sequential(*layers)
#         # 初始化
#         for layer in self.layers:
#             if isinstance(layer, nn.Linear):
#                 init.xavier_uniform_(layer.weight)
#                 if layer.bias is not None:
#                     init.zeros_(layer.bias)
#
#     def forward(self, x):
#         return self.layers(x)

class MHA(nn.Module):
    def __init__(self, Encoder_config, Decoder_config,FC1_config,FC2_config):
        super(MHA , self).__init__()
        self.Encoder_Net = Encoder(*Encoder_config)
        self.Decoder_Net = Decoder(*Decoder_config)
        self.FC1_Net = FC1(*FC1_config)
        self.FC2_Net = FC1(*FC2_config)
        # self.Trunk_Net = Trunk(*Trunk_config)

    def forward(self,Feature1,Feature2):
        # Auto-Encoder
        Encoder_output = self.Encoder_Net(Feature1)

        Decoder_output = self.Decoder_Net(Encoder_output)

        # Concentrated Froce Model
        FC1_output = self.FC1_Net(Feature2)

        FC2_output = self.FC2_Net(FC1_output)

        # #Trunk
        # Trunk_output = self.Trunk_Net(Trunk)
        #
        # #Combine
        # output = FC2_output @ Trunk_output.t()

        return FC2_output , Encoder_output , FC1_output , Decoder_output



