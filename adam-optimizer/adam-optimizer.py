import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """

    param, grad, m, v = map(np.asarray, (param, grad, m, v))
    
    m_raw = beta1 * m + (1 - beta1) * grad
    m_new = m_raw / (1 - beta1 ** t)
    
    v_raw = beta2 * v + (1 - beta2) * grad ** 2 
    v_new = v_raw / (1 - beta2 ** t)
    
    param_new = param - lr * m_new / (np.sqrt(v_new) + eps)

    return (param_new, m_raw, v_raw)