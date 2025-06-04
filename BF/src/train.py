# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午5:29
# @File : train.py
"""
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import data_loader
from tqdm import tqdm
from model import est_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_epoch = 2500
lr = 5e-5
RESULT_TRAIN = []
root = '../data/'
log_train = open('../log/train_log_fn_8.txt', 'w')

def train(model, n_epoch, training_data, loss_fn, optimizer):
    log_train.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    for epoch in tqdm(range(n_epoch)):
        total_loss_train = 0
        for batch, (data, labels) in enumerate(training_data):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            model.train()
            output = model(data)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            # print("Epoch: [{}/{}] | Batch: [{}/{}] | Loss: {:6f}".format(epoch, n_epoch, batch + 1, len(training_data), loss.item()))
        total_loss_train = total_loss_train / len(training_data)
        res_e = 'Epoch: [{}/{}], training loss: {:6f}'.format(epoch, n_epoch, total_loss_train)
        tqdm.write(res_e)
        log_train.write(res_e + '\n')
        RESULT_TRAIN.append([n_epoch, total_loss_train])
    return model


if __name__ == '__main__':
    torch.manual_seed(1)
    file_name = 'flownum_set.csv'
    data_path = root + file_name
    training_data = data_loader.training_data(path=data_path, batch_size=1)
    model = est_model().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    model = train(model=model, n_epoch=n_epoch, training_data=training_data, loss_fn=loss_fn, optimizer=optimizer)
    torch.save(model.state_dict(), "../checkpoints/model_fn_8.pth")
    log_train.close()
    res_train = np.asarray(RESULT_TRAIN)
    np.savetxt("../log/res_train.csv", res_train, fmt='%.6f', delimiter=",")
