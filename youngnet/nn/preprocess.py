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
