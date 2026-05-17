import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        temp = z - np.max(z)
        somme = np.exp(temp)
        return np.round(somme/np.sum(somme), 4)
