def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    for _ in range(steps):
        delx = 2*a*x0 + b
        x0 -= lr*delx

    return x0