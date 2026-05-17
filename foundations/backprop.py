import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = (x @ w) + b
        y_hat = 1/(1+ np.exp(-z))
        grad_loss = y_hat - y_true
        grad_w = grad_loss*y_hat*(1-y_hat)*x
        grad_b = grad_loss*y_hat*(1-y_hat)
        return (np.round(grad_w, 5), np.round(grad_b, 5))