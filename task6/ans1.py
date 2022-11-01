import numpy as np
from math import sqrt


class SelfAttention:
    def __init__(self, input_dim, dim_k, dim_v):
        self.linear_q = np.random.randn(input_dim, dim_k)
        self.linear_k = np.random.randn(input_dim, dim_k)
        self.linear_v = np.random.randn(input_dim, dim_v)
        self.factor = 1 / sqrt(dim_k)
    
    def __call__(self, x: np.ndarray):
        return self.forward(x)
    
    def forward(self, x: np.ndarray):
        Q = x @ self.linear_q
        K = x @ self.linear_k 
        V = x @ self.linear_v
        attention_weight = softmax(Q @ K.transpose(0,2,1) * self.factor, axis=-1)
        output = attention_weight @ V
        
        return output


def softmax(x: np.ndarray, axis=None, keepdims=True):
    """
    Softmax function
    """
    x_sub_max = x - x.max()     # 数值稳定性
    exp_x = np.exp(x_sub_max)
    sum_exp_x  = np.sum(exp_x, axis=axis, keepdims=keepdims)
    return exp_x / sum_exp_x


x = np.random.randn(50,30,20)
self_attention = SelfAttention(20, 40, 30)
y = self_attention(x)
print(y)
