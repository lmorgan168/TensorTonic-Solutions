def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    # update x = x - lr * derivative f'(x)

    x = x0
    
    for _ in range(steps):
        gradient = 2*a*x + b

        x = x - (lr * gradient)

    return x