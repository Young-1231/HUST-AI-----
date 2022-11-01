import sys
sys.path.append("..")
import numpy as np
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt 

from youngnet import nn 
from youngnet import functional as F
from youngnet import data_loader, onehot, standardization, classify_accuracy, train_test_split


X, y = load_digits(return_X_y=True)
y = onehot(y, 10)
# 数据集划分为6:1
train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size=len(X) // 7
)


train_X, test_X = standardization(train_X, test_X)

n_input = train_X.shape[1]
n_hiddens = [60, 100, 60, 30]
n_output = 10

class Net(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, p_dropout=None):
        super().__init__()
        self.p_dropout = p_dropout
        self.fc1 = nn.Linear(n_input, n_hiddens[0])
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(n_hiddens[0], n_hiddens[1])
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(n_hiddens[1], n_hiddens[2])
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(n_hiddens[2], n_hiddens[3])
        self.tanh4 = nn.Tanh()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        self.fc5 = nn.Linear(n_hiddens[3], n_output)
    
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        x = self.tanh3(x)
        x = self.fc4(x)
        x = self.tanh4(x)
        if self.p_dropout is not None:
            x = self.dropout(x)
        x = self.fc5(x)
        return x
    
    def backpropagation(self, grad: np.ndarray):
        grad = self.fc5.backpropagation(grad)
        if self.p_dropout is not None:
            grad = self.dropout.backpropagation(grad)
        grad = self.tanh4.backpropagation(grad)
        grad = self.fc4.backpropagation(grad)
        grad = self.tanh3.backpropagation(grad)
        grad = self.fc3.backpropagation(grad)
        grad = self.tanh2.backpropagation(grad)
        grad = self.fc2.backpropagation(grad)
        grad = self.tanh1.backpropagation(grad)
        grad = self.fc1.backpropagation(grad)
    


net = Net(n_input, n_hiddens, n_output, p_dropout=0.5)
nodes, grads = net.parameters()
optimizer = nn.SGD(nodes, grads, lr=0.1)

loss = nn.CrossEntropyLoss()

num_epochs = 50
batch_size = 32

loader = data_loader(train_X, train_y, batch_size, True)
loss_list, train_acc, test_acc = [], [], []


for epoch in range(num_epochs):
    net.train()
    train_loss = 0.
    
    for batch_x, batch_y in loader:
        output = net(batch_x)
        # 上一个iter累积梯度清零
        net.zero_grad()
        l, dy = loss(output, batch_y)
        net.backpropagation(dy)
        optimizer.step()

    net.eval()
    output = net(train_X)
    loss_list.append(loss(output, train_y)[0])

    train_acc.append(classify_accuracy(output, train_y))
    output = net(test_X)
    test_acc.append(classify_accuracy(output, test_y))

    if epoch % 10 == 9:
        print(f"epoch {epoch+1:3d}, train loss {loss_list[-1]:.6f}, train acc {train_acc[-1]:.4f}, test acc {test_acc[-1]:.4f}")


def visualize(train_acc_list, test_acc_list, loss_list):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label="training loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(test_acc_list, label="Test Accuracy")
    plt.legend()
    
    plt.savefig("./img/mlp_tanh_5layer_dropout.png")
    plt.show()


visualize(train_acc, test_acc, loss_list)
