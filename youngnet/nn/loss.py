import numpy as np

from .module import Module
from . import functional as F


class Loss(Module):
    """
    loss function的基类
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        assert self.reduction.lower() in {'mean', 'sum'}, "Invalid reduction"

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError

    def backpropagation(self, grad: np.ndarray):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        return F.mse_loss(y_pred, y_true, reduction=self.reduction)

    def backpropagation(self, grad: np.ndarray):
        return None


class NLLLoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        return F.nll_loss(y_pred, y_true)

    def backpropagation(self, grad: np.ndarray):
        return None


class CrossEntropyLoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        return F.cross_entropy_loss(y_pred, y_true)

    def backpropagation(self, grad: np.ndarray):
        return None 