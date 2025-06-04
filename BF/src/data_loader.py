# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午5:14
# @File : data_loader.py
"""
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def training_data(path, batch_size):
    train_data = pd.read_csv(path)
    train_X = train_data.iloc[:, 0].values
    train_y = train_data.iloc[:, 3].values
    train_X_t = torch.from_numpy(train_X.astype(np.float32))
    train_y_t = torch.from_numpy(train_y.astype(np.float32)).unsqueeze(-1)
    train_data_s  = Data.TensorDataset(train_X_t, train_y_t)
    train_data_loader = Data.DataLoader(
        dataset=train_data_s,
        batch_size=batch_size,
        shuffle=True
    )
    return train_data_loader

def testing_data(path, batch_size):
    test_data = pd.read_csv(path)
    test_X = test_data.iloc[:, 0:5].values
    test_y = test_data.label.values
    test_X_t = scaler.fit_transform(test_X)
    test_X_t = torch.from_numpy(test_X_t.astype(np.float32))
    test_y_t = torch.from_numpy(test_y.astype(np.float32)).unsqueeze(-1)
    test_data_s  = Data.TensorDataset(test_X_t, test_y_t)
    test_data_loader = Data.DataLoader(
        dataset=test_data_s,
        batch_size=batch_size,
        shuffle=True
    )
    return test_data_loader

