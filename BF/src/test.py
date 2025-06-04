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


RESULT_TEST = []
log_test = open('../log/test_log.txt', 'a')

def testing(model, testing_data, loss_fn):
    total_loss_test = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(testing_data):
            model.eval()
            pred = model(data)
            loss = loss_fn(pred, target)
            total_loss_test += loss.item()
        res = 'Test total loss: {:6f}'.format(total_loss_test)
    tqdm.write(res)
    log_test.write(res + '\n')
    RESULT_TEST.append([batch, total_loss_test])


if __name__ == '__main__':
    torch.manual_seed(1)
    data_path = ''
    testing_data = data_loader.testing_data(path=data_path, batch_size=512)
    model = est_model()
    loss_fn = nn.MSELoss()
    testing(model=model, testing_data=testing_data, loss_fn=loss_fn)
    log_test.close()
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_test.csv', res_test, fmt='%6f', delimiter=',')
