import numpy as np


class Calculator:
    def __init__(self, n_points=100):
        self.xx = np.linspace(0, 1, n_points)

    @property
    def step(self):
        return self.xx[1] - self.xx[0]

    def volume(self, points):
        return 4 * np.pi * np.sum(self.xx * points) * self.step

    def length(self, points):
        grad = (points[1:] - points[:-1]) / self.step
        return np.sum((1 + grad ** 2) ** 0.5) * self.step

    def ratio(self, points):
        return self.volume(points) / self.length(points) ** 3


if __name__ == "__main__":
    calc = Calculator()
    sphere = (1 - calc.xx ** 2) ** 0.5
    print(calc.volume(sphere))
    print(calc.length(sphere))
    print(calc.ratio(sphere))
