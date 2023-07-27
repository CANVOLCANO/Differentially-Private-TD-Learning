import numpy as np
import torch.nn as nn
import torch

class Agent(nn.Module):
    def __init__(self, in_features=40, mid_features=20, out_features=1, bias=False, param_range=[0,1]):
        super().__init__()
        self.linear1 = nn.Linear(in_features, mid_features, bias=bias)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(mid_features, out_features, bias=bias)
        self.param_range = param_range
        self.weight_initialization()

    @torch.no_grad()
    def weight_initialization(self):
        low = self.param_range[0]
        high = self.param_range[1]
        for param in self.parameters():
            torch.nn.init.xavier_uniform_(param, gain=1)

    @torch.no_grad()
    def print_params(self):
        idx = 0
        for param in self.parameters():
            print('idx:', idx)
            print(" shape:", param.shape)
            print(" max:", param.max())
            print(" min", param.min())
            idx += 1

    def forward(self, x):
        y = self.linear1(x)
        y = self.elu(y)
        y = self.linear2(y)    
        return y
    

class ValueAgent(Agent):
    """
    State -> Value
    """
    def __init__(self, in_features=2, mid_features=50, out_features=1, param_range=[0,1]):
        super().__init__(in_features=in_features, mid_features=mid_features, out_features=out_features,param_range=param_range)



        
        

    
        