import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from youngnet import nn
from youngnet.nn import Module, data_loader, train_test_split


kf = KFold(n_splits=10, shuffle=True)
data1 = pd.read_csv('../LR_dataset/data1.csv', header=None)
data2 = pd.read_csv('../LR_dataset/data2.csv', header=None)
data1 = data1.values
data2 = data2.values
data1_x = data1[:, :-1]
data1_y = nn.onehot(data1[:, -1].astype(int).reshape(-1), 2)
data2_x = data2[:, :-1]
data2_y = nn.onehot(data2[:, -1].astype(int).reshape(-1), 2)

# data1_x, _ = nn.standardization(data1_x, np.zeros_like(data1_x))
data2_x, _ = nn.standardization(data2_x, np.zeros_like(data2_x))
# data2_x, _ = nn.min_max_scale(data2_x, np.zeros_like(data2_x))

# train_data1_x, test_data1_x, train_data1_y, test_data1_y = train_test_split(data1_x, data1_y, len(data1_x) // 8)
# train_data2_x, test_data2_x, train_data2_y, test_data2_y = train_test_split(data2_x, data2_y, len(data2_x) // 8)
# train_data1_x, test_data1_x = nn.standardization(train_data1_x, test_data1_x)
# train_data1_x, test_data1_x = nn.min_max_scale(train_data1_x, test_data1_x)

# zscore标准化
# train_data2_x, test_data2_x = nn.standardization(train_data2_x, test_data2_x)
# train_data2_x, test_data2_x = nn.min_max_scale(train_data2_x, test_data2_x)


class LogisticRegression(Module):
    def __init__(self, n_input, threshold_score=0.5):
        super().__init__()
        self.threshold_score = threshold_score
        self.fc1 = nn.Linear(n_input, 2)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x : np.ndarray):
        x = self.fc1(x)
        x = self.sigmoid1(x)
        self.output = x
        return x

    def backpropagation(self, grad: np.ndarray):
        grad = self.sigmoid1.backpropagation(grad)
        grad = self.fc1.backpropagation(grad)

    def logistic_calc(self):
        logistic_result = np.zeros_like(self.output)
        logistic_result[self.output > self.threshold_score] = 1
        return logistic_result


model1 = LogisticRegression(n_input=data1_x.shape[1])
nodes, grads = model1.parameters()
optimizer1 = nn.SGD(nodes, grads, lr=0.01)
#optimizer1 = nn.Adam(nodes, grads, weight_decay=0.001)
num_epochs_1 = 500
batch_size_1 = 64
loss = nn.MSELoss()
loss_list_1, train_acc_1, test_acc_1 = [], [], []
loss_list_2, train_acc_2, test_acc_2 = [], [], []

model2 = LogisticRegression(n_input=data2_x.shape[1])
nodes, grads = model2.parameters()
optimizer2 = nn.SGD(nodes, grads, weight_decay=0.01)
# optimizer2 = nn.Adam(nodes, grads, weight_decay=0.01)

num_epochs_2 = 500
batch_size_2 = 64


def train(model: LogisticRegression, x, y, num_epochs, loss_func, optimizer, loss_list, train_acc_list, test_acc_list, kf: KFold):
    count = 0
    for train_index, test_index in kf.split(x):
        train_x, test_x = x[train_index], x[test_index]
        train_y, test_y = y[train_index], y[test_index]

        batch_size = len(train_x)
        loader = data_loader(train_x, train_y, batch_size, True)

        print(f"Fold: {count}")
        count += 1
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.

            for batch_x, batch_y in loader:
                output = model(batch_x)
                model.zero_grad()
                l, dy = loss_func(output, batch_y)
                model.backpropagation(dy)
                optimizer.step()

            model.eval()
            output = model(train_x)
            loss_list.append(loss_func(output, train_y)[0])

            train_acc_list.append(nn.classify_accuracy(model.logistic_calc(), train_y))
            output = model(test_x)
            test_acc_list.append(nn.classify_accuracy(model.logistic_calc(), test_y))

            if (epoch + 1) % 20 == 0:
                print(
                    f"epoch {epoch + 1:3d}, train loss {loss_list[-1]:.6f}, train acc {train_acc_list[-1]:.4f}, test acc {test_acc_list[-1]:.4f}")


def visualize(train_acc_list, test_acc_list, loss_list):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label="training loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.legend()

    plt.savefig("./img/10fold_data2_lr_zscore_gd.png")
    plt.show()


# train(model1, data1_x, data1_y, num_epochs_1, loss, optimizer1, loss_list_1, train_acc_1, test_acc_1, kf)
# visualize(train_acc_1, test_acc_1, loss_list_1)

train(model2, data2_x, data2_y, num_epochs_2, loss, optimizer2, loss_list_2, train_acc_2, test_acc_2, kf)
visualize(train_acc_2, test_acc_2, loss_list_2)
# train(model1, data1_x, data1_y, num_epochs_1, loss, optimizer1, loss_list_1, train_acc_1, test_acc_1, kf)
# visualize(train_acc_1, test_acc_1, loss_list_1)
