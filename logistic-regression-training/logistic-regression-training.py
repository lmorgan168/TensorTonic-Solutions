import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
    
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # derivative of sigmoid(z) = sigmoid(z)(1-sigmoid(z))
    # Derivative of Loss function with respect to sigmoid(z)
    # dL/dp = - (y * 1/p - (1-y)(1/1-p)))
    # Dp/dz = p ( 1 - p)
    # chain rule = -(y/p - (1-y)/(1-p)) * (p) ( 1-p) = -((y * 1-p) - (1-y) * p) = - (y - py - p + py)
    # = p -y

    # chain rule the loss function one more time for respect to w and respect to b
    # dl/dw = p-y * x
    # dl/db = p-y*1

    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        error = p-y
        dw = (1/N) * (X.T @ error)
        db = (1/N) * np.sum(error)

        w -= lr * dw
        b -= lr * db
    return w,b