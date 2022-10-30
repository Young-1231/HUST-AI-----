from . import init
from . import functional as F
from .module import Module, Parameter, Grad
from .utils import process_grad
import numpy as np
from typing import Dict, Tuple


class Optimizer:
    def __init__(self, nodes: Dict, grads: Dict):
        self.nodes = nodes
        self.grads = grads

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, nodes: Dict, grads: Dict, lr: float = 1e-3, momentum: float = 0.8, weight_decay: float = 0.):
        super().__init__(nodes, grads)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {k: np.zeros_like(node) for k, node in self.nodes.items()}

    def step(self):
        for key in self.nodes.keys():
            grad = self.grads[key] + self.weight_decay * self.nodes[key]
            self.v[key] *= self.momentum
            self.v[key] += self.lr * grad
            self.nodes[key] -= self.v[key]


class Adam(Optimizer):
    def __init__(self, nodes: Dict, grads: Dict, lr: float = 1e-3, betas: Tuple[float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0):
        super().__init__(nodes, grads)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {key: np.zeros_like(node) for key, node in self.nodes.items()}
        self.v = {key: np.zeros_like(node) for key, node in self.nodes.items()}
        self.t = 1

    def step(self):
        for key in self.nodes.keys():
            grad = self.grads[key] + self.weight_decay * self.nodes[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2
            m_t = self.m[key] / (1 - self.beta1 ** self.t)
            v_t = self.v[key] / (1 - self.beta2 ** self.t)
            self.nodes[key] -= self.lr * m_t / (v_t ** 0.5 + self.eps)
        self.t += 1
