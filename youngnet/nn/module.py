import numpy as np
from collections import OrderedDict
from typing import List

class Grad:
    def __init__(self, data: np.ndarray):
        self.data = data
    
    def __repr__(self):
        return f"Grad: \n{self.data}"


class Parameter:
    def __init__(self, data: np.ndarray):
        self.data = data
        
    def __repr__(self):
        return f"Parameter: \n{self.data}"


class Module:
    def __init__(self):
        self._train = True
        self._parameters = OrderedDict()
        self._grads = OrderedDict()
        self.input = None
        self.output = None
    
    def __call__(self, *x):
        return self.forward(*x)
    
    def __setattr__(self, __name: str, __value):
        self.__dict__[__name] = __value
        if isinstance(__value, Parameter):
            self._parameters[__name] = __value.data
        if isinstance(__value, Grad):
            self._grads[__name.split("_")[0]] = __value.data
        if isinstance(__value, Module):
            for key in __value._parameters:
                self._parameters[__name + "." + key] = __value._parameters[key]
            for key in __value._grads:
                self._grads[__name + "." + key.split("_")[0]] = __value._grads[key]
                
    def init_grad(self):
        for key, value in self._parameters.items():
            self._grads[key] = np.zeros_like(value)
    
    def __repr__(self) -> str:
        module_list = [module for module in self.__dict__.items() if isinstance(module[1], Module)]
        return "{}(\n{}\n)".format(self.__class__.__name__,
                "\n".join(["{:>10} : {}".format(module_name, module) for module_name, module in module_list]))

    def parameters(self):
        return self._parameters, self._grads

    def train(self, mode: bool = True):
        self.set_module_state(mode)
    
    def set_module_state(self, mode):
        self._train = mode
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.set_module_state(mode)
    
    def forward(self, x : np.ndarray):
        raise NotImplementedError

    def backpropagation(self, grad: np.ndarray):
        raise NotImplementedError

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        """梯度清零"""
        for key, _ in self._grads.items():
            self._grads[key] *= 0.

    @property
    def state_dict(self):
        return self._parameters

#TODO: 待完善
    def load_state_dict(self, state_dict: 'OrderedDict[str, np.ndarray]', strict: bool = True):
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        state_dict = state_dict.copy()
        for k in self._parameters.keys():
            if k in state_dict.keys():
                self._parameters[k] = state_dict[k]
            else:
                missing_keys.append(k)

        if len(missing_keys) != 0:
            raise ValueError("权重文件中缺少项")

