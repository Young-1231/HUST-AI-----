from . import init
from . import functional as F
from .module import Module

import numpy as np


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray):
        self.input = x
        self.output = F.relu(x)
        return self.output

    def backpropagation(self, grad: np.ndarray):
        cur_grad = F.relu_backward(self.input, grad)
        return cur_grad

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray):
        self.output = F.sigmoid(x)
        return self.output

    def backpropagation(self, grad: np.ndarray):
        cur_grad = F.sigmoid_backward(self.output, grad)
        return cur_grad

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray):
        self.input = x

        self.output = F.tanh(x)
        return self.output

    def backpropagation(self, grad: np.ndarray):
        cur_grad = F.tanh_backward(self.output, grad)
        return cur_grad
