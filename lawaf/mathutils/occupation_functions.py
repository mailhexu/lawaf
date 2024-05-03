import numpy as np
from scipy.special import erfc
from functools import partial

__all__ = ["occupation_func"]

def unity_func(x, *args):
    return np.ones_like(x, dtype=float)

def fermi_func(x, mu=0.0, sigma=1.0):
    return 0.5 * erfc((x - mu) / sigma)

def gauss_func(x, mu=0.0, sigma=1.0):
    return np.exp(-1.0 * (x - mu) ** 2 / sigma**2)

def window_func(x, lower=-1, upper=1, sigma=0.01):
    return 0.5 * erfc((x - lower) / sigma) - 0.5 * erfc((x - upper) / sigma)

def linear_func(x):
    return x




def occupation_func(ftype=None, *args):
    """
    Return a Weight function.
    """
    funcdict = {
        "unity": unity_func,
        "Fermi": fermi_func,
        "Gauss": gauss_func,
        "window": window_func,
        "linear": linear_func,
    }
    if ftype is None:
        return unity_func
    elif ftype in funcdict:
        return lambda x: funcdict[ftype](x, *args)
    else:
        raise ValueError(f"Unknown function type: {ftype}")
