import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x = np.asarray(x)
    res = 1/(1 + np.exp(-x))

    if(res.ndim == 0):
        return res.item()
    return res