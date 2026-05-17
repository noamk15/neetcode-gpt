class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        actual = init
        for _ in range(iterations):
            derivate = 2*actual
            actual = actual - (learning_rate * derivate)
        return round(actual, 5)
