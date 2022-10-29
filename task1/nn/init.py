from numpy.random import uniform, normal
import math
import numpy as np



def calculate_gain(activation: str, param: float = None):
    activation = activation.lower()
    recommendation_gain_dict = {
        "linear": 1,
        "conv1d": 1,
        "conv2d": 1,
        "sigmoid": 1,
        "tanh": 5 / 3,
        "relu": math.sqrt(2.),
        "leaky_relu":
        math.sqrt(2. / (1 + (param if param != None else 0.01)**2))
    }
    return recommendation_gain_dict[activation]



def _calculate_fan(parameter: np.ndarray):
    assert parameter.ndim >= 2
    fan_in, fan_out = parameter.shape[:2]
    if parameter.ndim > 2:
        receptive_field_size = math.prod(parameter.shape[2:])
        fan_in *= receptive_field_size
        fan_out *= receptive_field_size
    return fan_in, fan_out


def uniform_(parameter: np.ndarray, a=0., b=1.):
    parameter = uniform(a, b, parameter.shape)
    return parameter


def normal_(parameter: np.ndarray, mean=0, std=1.):
    parameter = normal(mean, std, size=parameter.shape)
    return parameter


def xavier_uniform_(parameter: np.ndarray, gain: float=1.):
    fan_in, fan_out = _calculate_fan(parameter)
    bound = gain * math.sqrt(6. / (fan_in + fan_out))
    return uniform_(parameter, -bound, bound)


def xavier_normal_(parameter: np.ndarray, gain: float=1.):
    fan_in, fan_out = _calculate_fan(parameter)
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return normal_(parameter, std=std)


def kaiming_normal_(parameter: np.ndarray, a:float = 0, mode = "fan_in", activation='relu'):
    fan_in, fan_out = _calculate_fan(parameter)
    fan = { "fan_in":fan_in, "fan_out":fan_out}[mode]
    gain = calculate_gain(activation, a)
    std = gain / math.sqrt(fan)
    return normal_(parameter, std=std)
