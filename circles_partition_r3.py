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

    def min_origin_dist(self):
        # todo :: consider more optimal calculation ^^
        # (i.e. can do this just with linear algebra without discretizing
        # to points, but doing this quickly to be lazy)
        return np.min(np.linalg.norm(self.get_points(n_points=20)))


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


def _sphere_circle_arr_intersection(r):
    """
    intersection of sphere radius r in the origin with the set of circles
    centered on (4n+1, 0, 0)
    """
    n = (r + 2) // 4 * (-1 if (r // 2) % 2 else 1)
    print(r, n)
    circ_x0 = 4 * n + 1
    x = (r ** 2 + circ_x0 ** 2 - 1) / (2 * circ_x0)
    y2 = r ** 2 - x ** 2
    if y2 < 0:
        return [(0, 0), (0, 0)]
    return [(x, y2 ** 0.5), (x, -y2 ** 0.5)]


def animate():
    circs = []
    for i in range(-3, 3):
        circs.append(
            CircleIn3D(
                position=(4 * i + 1, 0, 0),
                normal=(0, 0, 1),
                radius=1,
            )
        )
    for origin_sph_r in np.linspace(0, 30, 30)[1:]:  # skip 0
        # calculate intersections w circle:
        pts = _sphere_circle_arr_intersection(origin_sph_r)
        # todo
        # initial theta depends on gradient?
        # also these will come in pairs
        # todo :: deal with exactly on axis ones separately
        for pt_x, pt_y in pts:
            grad_min = pt_y / pt_x
            for log_grad in np.linspace(0, 10, 10):
                grad = grad_min * np.exp(log_grad)
                vx = np.cos(grad)
                vy = np.sin(grad)
                # todo :: do the rest of the partition :)
                circs.append(
                    CircleIn3D(
                        # todo temp nonsense
                        position=(vx * origin_sph_r, vy * origin_sph_r, 0),
                        normal=(vx, vy, 0),
                        radius=origin_sph_r * (np.tanh(grad - grad_min) + 0.01),
                    )
                )
    anim = ManyCircleAnimator(
        circles=sorted(
            circs,
            key=CircleIn3D.min_origin_dist,
        )
    )
    anim.plot()


if __name__ == "__main__":
    animate()
