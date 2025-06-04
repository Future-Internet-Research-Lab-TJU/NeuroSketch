import torch
from torch.utils.data import DataLoader
import numpy
import pandas as pd
import data_loader
from model import est_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

counter0 = []
counter1 = []
counter2 = []
counter3 = []
counter4 = []
counter5 = []
preds = []
groundtruth = []


def testing(model, testing_data):
    with torch.no_grad():
        for batch, (X, y) in enumerate(testing_data):
            model.eval()
            X, y = X.to(device), y.to(device)

            # nor_X = (X - train_X_min) / (train_X_max - train_X_min)

            pred = model(X)
            # pred = model(nor_X)
            y = y.view(-1,1)
            # inv_pred = torch.round(pred * (train_y_max - train_y_min) + train_y_min)

            X = X.cpu().numpy()
            # inv_pred = inv_pred.cpu().numpy()
            inv_pred = pred.cpu().numpy()
            y = y.cpu().numpy()

            for x in X:
                counter0.append(x[0])
                counter1.append(x[1])
                counter2.append(x[2])
                # counter3.append(x[3])
                # counter4.append(x[4])
                # counter5.append(x[5])
            
            preds.extend(inv_pred.reshape([-1]))
            
            groundtruth.extend(y.reshape([-1]))

    results = {
        'counter0':counter0,
        'counter1':counter1,
        'counter2':counter2,
        # 'counter3':counter3,
        # 'counter4':counter4,
        # 'counter5':counter5,
        'groundtruth':groundtruth,
        'prediction':preds}

    res_df = pd.DataFrame(results)
    res_df.to_csv('BFCM.csv')

if __name__ == '__main__':
    torch.manual_seed(1)
    data_path = '../test_data/dateset2.csv'
    # _, train_X_min, train_X_max, train_y_min, train_y_max = data_loader.training_data("../test_data/dateset1.csv", batch_size=8192)
    _, testing_data = data_loader.testing_data(path=data_path, batch_size=300000)
    model = est_model().to(device)
    model.load_state_dict(torch.load('../checkpoints/model_6.pth'))
    testing(model=model, testing_data=testing_data)