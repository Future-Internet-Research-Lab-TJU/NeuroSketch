# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/5/7 下午7:04
# @File : extract_param.py
"""
import torch
from model import est_model
import numpy as np
model = est_model()
model.load_state_dict(torch.load("../checkpoints/model_hh.pth"))

# model.load_state_dict()
# print(net_load[0])
print(list(model.parameters()))
np.savetxt("../parameters/0529/w1_hh.csv", list(model.parameters())[0].detach().numpy(), fmt='%.6f', delimiter=',')
np.savetxt("../parameters/0529/b1_hh.csv", list(model.parameters())[1].detach().numpy(), fmt='%.6f', delimiter=',')
np.savetxt("../parameters/0529/w2_hh.csv", list(model.parameters())[2].detach().numpy(), fmt='%.6f', delimiter=',')
np.savetxt("../parameters/0529/b2_hh.csv", list(model.parameters())[3].detach().numpy(), fmt='%.6f', delimiter=',')