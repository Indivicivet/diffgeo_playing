import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt


class Calculator:
    def __init__(self, n_points=100):
        self.xx = np.linspace(0, 1, n_points)

    @property
    def step(self):
        return self.xx[1] - self.xx[0]

    def volume(self, points):
        return 4 * np.pi * np.sum(self.xx * (points - points[-1])) * self.step

    def length(self, points):
        grad = (points[1:] - points[:-1]) / self.step
        return (
            np.sum((1 + grad ** 2) ** 0.5)
            + (1 + points[-1] ** 2) ** 0.5  # final term penalty
        ) * self.step

    def ratio(self, points):
        return self.volume(points) / self.length(points) ** 3


if __name__ == "__main__":
    calc = Calculator()
    sphere = (1 - calc.xx ** 2) ** 0.5
    print(calc.volume(sphere))  # 4 pi / 3 ~ 4.1888
    print(calc.length(sphere))  # pi / 2 ~ 1.5708
    print(calc.ratio(sphere))  # 32 / (3 pi ^2) ~ 1.0808

    # clear upper bound is sqrt(2) pi / 3 ~ 1.481
    res = optimize.minimize(
        lambda x: -calc.ratio(x),
        x0=sphere,
    )
    print(res)
    print(calc.volume(res.x))
    print(calc.length(res.x))
    print(calc.ratio(res.x))
    plt.plot(calc.xx, res.x)
    plt.show()
