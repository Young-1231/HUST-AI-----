from matplotlib.pyplot import axis
import numpy as np
import pdb


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
    affine = x @ weight
    if bias is not None:
        affine = affine + bias
    return affine


def linear_backward(weight: np.ndarray, input: np.ndarray, grad: np.ndarray):
    """
    全连接层的反向传播
    
    """
    N = input.shape[0]
    dz = grad @ weight.T
    
    dw = input.T @ grad
    db = np.sum(grad, axis=0)
    return {"weight": dw / N, "bias": db / N, "cur_grad": dz}


def sigmoid(x: np.ndarray):
    sigmoid_result = np.zeros(x.shape)
    sigmoid_result[x.data > 0] = 1 / (1 + np.exp(-x[x > 0]))
    sigmoid_result[x.data <= 0] = 1 - 1 / (1 + np.exp(x <= 0))
    return sigmoid_result


def sigmoid_backward(output: np.ndarray, grad: np.ndarray):
    return output * (1 - output) * grad


def relu(x: np.ndarray):
    return np.maximum(0, x)


def relu_backward(x: np.ndarray, grad: np.ndarray):
    dz = np.where(np.greater(x, 0), grad, 0)
    return dz


def softmax(x: np.ndarray, axis=None, keepdims=False):
    # 考虑数值稳定性，减去最大值
    x_sub_max = x - x.max()
    exp_x = np.exp(x_sub_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=keepdims)


def mse_loss(y_pred, y_true, reduction='mean'):
    """
    Mean Squared Loss
    """
    squared_sum = np.square(y_pred - y_true)
    dy = y_pred - y_true
    reduction = reduction.lower()
    assert type(reduction) is str
    if reduction == 'mean':
        return np.mean(squared_sum), dy
    elif reduction == 'sum':
        return np.sum(squared_sum), dy
    else:
        raise ValueError('Invalid reduction(reduction must be \'mean\' or \'sum\')')


def nll_loss(y_pred, y_true, reduction='mean'):
    """
    负对数似然
    """
    nll = -y_pred * y_true
    reduction = reduction.lower()
    assert type(reduction) is str
    if reduction == 'mean':
        return np.mean(nll)
    elif reduction == 'sum':
        return np.sum(nll)
    else:
        raise ValueError('Invalid reduction(reduction must be \'mean\' or \'sum\')')



