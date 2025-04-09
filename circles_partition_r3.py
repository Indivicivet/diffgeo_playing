from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as mpl_animation


@dataclass
class CircleIn3D:
    position: np.ndarray
    normal: np.ndarray
    radius: float

    def __post_init__(self):
        self.position = np.array(self.position)
        self.normal = np.array(self.normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)

    def get_points(self, n_points=100):
        not_parallel = np.array(
            [1, 0, 0]
            if np.allclose(self.normal, [0, 0, 1])
            else [0, 0, 1]
        )
        v0 = np.cross(self.normal, not_parallel)
        v0 /= np.linalg.norm(v0)
        v1 = np.cross(self.normal, v0)  # normalized as normal is
        t = np.linspace(0, 2 * np.pi, n_points)
        return (
            self.position[:, np.newaxis]
            + self.radius * (np.outer(v0, np.cos(t)) + np.outer(v1, np.sin(t)))
        )


@dataclass
class ManyCircleAnimator:
    circles: Sequence[CircleIn3D]
    latest_idx: int = 0
    lines: Sequence = field(default_factory=list)
    update_rate: float = 10
    _ax = None

    def _update(self, frame):
        if self.latest_idx < len(self.circles) and self._ax is not None:
            self.lines.append(
                self._ax.plot(*self.circles[self.latest_idx].get_points())[0]
            )
            self.latest_idx += 1
        return self.lines

    def plot(self):
        fig = plt.figure()
        self._ax = fig.add_subplot(111, projection="3d")
        _anim = mpl_animation.FuncAnimation(
            fig,
            self._update,
            interval=1000 / self.update_rate,
        )
        plt.show()


def animate():
    circs = []
    for i in range(-10, 10):
        circs.append(
            CircleIn3D(
                position=(4 * i + 1, 0, 0),
                normal=(0, 0, 1),
                radius=1,
            )
        )
    anim = ManyCircleAnimator(circles=circs)
    anim.plot()


if __name__ == "__main__":
    animate()
