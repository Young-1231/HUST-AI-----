from .module import Module 
import numpy as np


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = p 
    
    def forward(self, x: np.ndarray):
        if self.train:
            return x * np.random.binomial(1, 1 - self.p, x.shape[-1])
        return x * (1 - self.p)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
    