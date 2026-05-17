import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = (x @ w) + b
        if activation == "sigmoid":
            z = 1/(1 + np.exp(-z))
        else:
            z = np.maximum(0, z)
        return np.round(z, 5)
