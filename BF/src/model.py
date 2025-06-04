# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午5:14
# @File : model.py
"""
import torch.nn as nn
import torch.nn.functional as F

class est_model(nn.Module):
    def __init__(self, input_size=1, hidden_size=8, output_size=1):
        super(est_model, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = F.relu(self.hidden_layer_1(x))
        output = self.output_layer(output)
        return output