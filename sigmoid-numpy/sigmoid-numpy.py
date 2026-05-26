import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    # also support lists
    x = np.array(x)
    output = 1/(1+np.exp(-x))
    return output