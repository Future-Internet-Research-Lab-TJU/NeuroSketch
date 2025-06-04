# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午6:23
# @File : test.py
"""
import torch
import data_loader
from model import est_model


RESULT_TEST = []
# log_test = open('../log/test_log.txt', 'a')

def testing(model, testing_data):
    total_loss_test = 0
    ARE = 0
    AAE = 0
    flownum = 0
    results = []  # 新增：用于存储预测值和真实值
    with torch.no_grad():
        for batch, (data, target) in enumerate(testing_data):
            model.eval()
            pred = model(data)
            pred = torch.where(pred <= 0, torch.tensor(1.0), pred)
            flownum += len(data)
            ARE += torch.sum(abs((pred - target)) / target, dim=0).item()
            AAE += torch.sum(abs(pred - target), dim=0).item()
            # 新增：收集预测值和真实值
            for p, t in zip(pred, target):
                results.append([p.item(), t.item()])
        ARE = ARE / flownum
        AAE = AAE / flownum
    with open('flow_size.csv', 'a') as f:
        f.write(f"{ARE},{AAE}\n")
    print('{:6f}'.format(ARE))
    print('{:6f}'.format(AAE))


if __name__ == '__main__':
    torch.manual_seed(1)
    model = est_model()
    model.load_state_dict(torch.load('../checkpoints/model_3_d1.pth'))
    for i in range(1, 61):
        data_path = '../test_data/' + 'dateset' + str(2) +  '.csv'
        _, testing_data = data_loader.testing_data(path=data_path, batch_size=300000)
        testing(model=model, testing_data=testing_data)
    # for i in range(5, 11):
    #     data_path = 'dateset' + str(i) +  '.csv'
    #     packet_num, testing_data = data_loader.testing_data(path=data_path, batch_size=300000)
    #     testing(model=model, testing_data=testing_data)