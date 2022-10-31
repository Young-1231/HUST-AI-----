import numpy as np


def standardization(train_x, test_x):

    for i in range(train_x.shape[-1]):
        mean = train_x[:, i].reshape(-1).mean(axis=0)
        std = np.std(train_x[:, i].reshape(-1), axis=0)
        if mean == 0 or std == 0:
            continue
        train_x[:, i] -= mean
        train_x[:, i] /= std
        test_x[:, i] -= mean
        test_x[:, i] /= std

    return train_x, test_x


def min_max_scale(train_x, test_x):
    for i in range(train_x.shape[-1]):
        max = train_x[:, i].reshape(-1).max()
        min = train_x[:, i].reshape(-1).min()
        train_x[:, i] = (train_x[:, i] - min) / (max - min)
        test_x[:, i] = (test_x[:, i] - min) / (max - min)
    return train_x, test_x
