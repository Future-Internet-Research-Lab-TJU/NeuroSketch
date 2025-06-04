# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午5:14
# @File : model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class est_model(nn.Module):
    def __init__(self, input_size=6, hidden_size=50, output_size=1):
        super(est_model, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, 50)
        self.output_layer = nn.Linear(50, output_size)

    def forward(self, x):
        output = F.relu(self.hidden_layer(x))
        output = F.relu(self.hidden_layer_2(output))
        output = self.output_layer(output)
        return output