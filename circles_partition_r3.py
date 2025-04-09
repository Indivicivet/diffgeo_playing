from dataclasses import dataclass

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


def plot_time_t():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    circs = []
    for i in range(-10, 10):
        circs.append(
            CircleIn3D(
                position=(4 * i + 1, 0, 0),
                normal=(0, 0, 1),
                radius=1,
            )
        )
    latest_idx = 0
    lines = []

    def update(frame):
        nonlocal latest_idx
        new_circ = circs[latest_idx]
        latest_idx += 1
        line = ax.plot(*new_circ.get_points())[0]
        lines.append(line)
        return lines

    _anim = mpl_animation.FuncAnimation(fig, update, interval=1000)
    plt.show()


if __name__ == "__main__":
    plot_time_t()
