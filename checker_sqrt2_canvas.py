import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 2
SPACING_CONSTANT = 0.2

FORCE_CLIP = 1


def force_between(pt, other, is_same) -> np.ndarray:
    """
    force d/x^2 - 1/x
    """
    delta = other - pt
    dist = (delta**2).sum() ** 0.5
    if dist == 0:
        print(pt, other, is_same, "dist is 0???")  # todo :: exception?
        return np.array([0, 0])
    force_weight = (1 / SPACING_CONSTANT - [1, 2**0.5][is_same] / dist) / dist
    force_weight = abs(force_weight) ** 0.5 * np.sign(force_weight)
    # print(force_weight)
    return (delta / dist) * np.clip(force_weight, -FORCE_CLIP, FORCE_CLIP)


def evolve(aa, bb, delta_t):
    forces_aa = np.zeros_like(aa)
    forces_bb = np.zeros_like(bb)
    for i, pt0 in enumerate(aa):
        forces_aa[i] += sum(
            force_between(pt0, pt1, is_same=True) for j, pt1 in enumerate(aa) if i != j
        ) + sum(force_between(pt0, pt1, is_same=False) for pt1 in bb)
    for i, pt0 in enumerate(bb):
        forces_bb[i] += sum(
            force_between(pt0, pt1, is_same=True) for j, pt1 in enumerate(bb) if i != j
        ) + sum(force_between(pt0, pt1, is_same=False) for pt1 in aa)
    return aa + delta_t * forces_aa, bb + delta_t * forces_bb


if __name__ == "__main__":
    rng = np.random.default_rng(seed=23405)

    pts_a = rng.uniform(low=-1, high=1, size=[N, 2])
    pts_b = rng.uniform(low=-1, high=1, size=[N, 2])

    plt.scatter(*pts_a.T, label="pts a 0")
    plt.scatter(*pts_b.T, label="pts b 0")

    PLOT_ITERS = [50]

    for iter in tqdm(range(1, max(PLOT_ITERS) + 1)):
        pts_a, pts_b = evolve(pts_a, pts_b, delta_t=0.01)
        if iter in PLOT_ITERS:
            plt.scatter(*pts_a.T, label=f"pts a {iter}")
            plt.scatter(*pts_b.T, label=f"pts b {iter}")

    plt.legend()
    plt.show()
