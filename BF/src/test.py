# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午6:23
# @File : test.py
"""
import torch
import torch.nn as nn
import numpy as np
import data_loader
from tqdm import tqdm
from model import est_model


def testing(model, testing_data, loss_fn):
    total_loss_test = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(testing_data):
            model.eval()
            pred = model(data)
            loss = loss_fn(pred, target)
            total_loss_test += loss.item()


if __name__ == '__main__':
    torch.manual_seed(1)
    data_path = ''
    testing_data = data_loader.testing_data(path=data_path, batch_size=512)
    model = est_model()
    loss_fn = nn.MSELoss()
    testing(model=model, testing_data=testing_data, loss_fn=loss_fn)
