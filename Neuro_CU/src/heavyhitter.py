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
log_test = open('../log/test_log.txt', 'a')

def testing(model, testing_data, packet_num):
    real_hitter = {}
    detect_hitter = {}
    cnt = 0
    tp = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(testing_data):
            model.eval()
            mask = target >= 0.0005 * packet_num
            if mask.any():
                selected_data = data[mask.squeeze()]
                selected_target = target[mask.squeeze()]
                for d, t in zip(selected_data, selected_target):
                    cnt += 1
                    flowID = '-'.join(map(str, d.numpy().flatten().tolist()))
                    real_hitter[flowID] = t.item()
            pred = model(data)
            pred_mask = pred >= 0.0005 * packet_num
            if pred_mask.any():
                selected_data = data[pred_mask.squeeze()]
                selected_pred = pred[pred_mask.squeeze()]
                for d, p in zip(selected_data, selected_pred):
                    flowID = '-'.join(map(str, d.numpy().flatten().tolist()))
                    detect_hitter[flowID] = p.item()
                    
    for key in detect_hitter.keys():
        if key in real_hitter:
            tp += 1
    precision = tp / len(detect_hitter) if len(detect_hitter) > 0 else 0
    recall = tp / cnt if cnt > 0 else 0
    with open('precision_recall.csv', 'a') as f:
        f.write(f"{precision},{recall}\n")
    print('{:6f}'.format(precision))
    print('{:6f}'.format(recall))
    print('-------------------------------')


if __name__ == '__main__':
    torch.manual_seed(1)
    model = est_model()
    model.load_state_dict(torch.load('../checkpoints/modelbfcu_6_d2.pth'))
    for i in range(1, 61):
        data_path = '../test_data/dateset' + str(i) + '.csv'
        packet_num, testing_data = data_loader.testing_data(path=data_path, batch_size=300000)
        testing(model=model, testing_data=testing_data, packet_num=packet_num)

    # for i in range(5, 11):
    #     data_path = 'datesetweb.csv'
    #     packet_num, testing_data = data_loader.testing_data(path=data_path, batch_size=300000)
    #     testing(model=model, testing_data=testing_data, packet_num=packet_num)
