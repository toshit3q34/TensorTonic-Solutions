import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_sample, n_features = X.shape
    W = np.zeros(n_features)
    b = 0.0
    for _ in range(steps):
        z = np.dot(X,W) + b
        y_pred = _sigmoid(z)
        dW = (1/n_sample) * np.dot(X.T,(y_pred - y))
        db = np.mean(y_pred - y)
        W -= lr*dW
        b -= lr*db
    return (W,float(b))