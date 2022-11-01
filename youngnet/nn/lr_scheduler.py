from math import cos, pi
import numpy as np
from .optim import Optimizer


class Scheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.initial_lr = self.optimizer.lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        self.optimizer.lr = lr

    # 待重载的学习率调整方法
    def get_lr(self):
        raise NotImplementedError


class StepLR(Scheduler):
    """
    按固定步长衰减的学习率调整策略
    """
    def __init__(self, optimizer: Optimizer, step_size: int, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        return self.optimizer.lr
