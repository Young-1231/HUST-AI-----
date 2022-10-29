from . import init
from . import funcitonal as F
from .module import Module, Parameter, Grad
from .utils import process_grad

import math
import numpy as np


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.empty((self.in_features, self.out_features)))
        self.bias = Parameter(np.empty((1, self.out_features)))
        self.weight_grad = Grad(np.empty_like(self.weight.data))
        self.bias_grad = Grad(np.empty_like(self.bias.data))
        # 对参数进行初始化
        self.set_parameters()

    def set_parameters(self):
        self.weight.data = init.kaiming_normal_(self.weight.data, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.bias.data)
            bound = 1 / math.sqrt(fan_in)
            self.bias.data = init.uniform_(self.bias.data , -bound, bound)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

    def forward(self, x: np.ndarray):
        self.input = x
        self.output = F.linear(x, self.weight.data, self.bias.data)
        return self.output

    def backpropagation(self, next_grad):
        # 手动反向传播
        grad_dict = F.linear_backward(self.weight.data, self.input, next_grad)
        #add_weight_grad = process_grad(grad_dict['weight'], self.weight_grad.data)
        #add_bias_grad = process_grad(grad_dict['bias'], self.bias_grad.data
        self.weight_grad.data += grad_dict['weight']
        self.bias_grad.data += grad_dict['bias']
        return grad_dict['cur_grad']


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
    
    def backpropagation(self, grad: np.ndarray):
        cur_grad = F.sigmoid_backward(self.output, grad)
        return cur_grad

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)
