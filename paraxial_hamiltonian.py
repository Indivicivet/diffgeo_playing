import colorsys

import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm


VIZ_SAMPLES = 512
Z_SAMPLES = 600
VIEW_ONE_IN_N = 4
ANIM_TIME_S = 6
VIZ_QX, VIZ_QY = np.meshgrid(
    np.linspace(-1, 1, VIZ_SAMPLES),
    np.linspace(-1, 1, VIZ_SAMPLES),
)
VIZ_POINT_SIZE_Q = 0.01
ZS = np.linspace(0, 10, Z_SAMPLES)

# pseudo-enum
# todo :: we don't actually use this, it's just here to look pretty
AIR = 0
SOME_GLASS = 1

SOME_GLASS_RI = 1.5

T0 = 1  # air thick
T1 = 3  # glass thick
C0 = 0.2  # curvature coeff front face
C1 = -0.2  # curvature coeff back face

# ri:
# 1 (z < t0 + c0|q|^2)
# 1.5 (t0 + c0|q|^2 <= z < t0 + t1 + c1|q|^2)
# 1 (z > t0 + t1 + c1|q|^2)


def glass_type_viz(z):
    r_squared = VIZ_QX ** 2 + VIZ_QY ** 2
    return np.logical_and(
        T0 + C0 * r_squared <= z,
        z <= T0 + T1 + C1 * r_squared,
    )


def glass_type_to_picture(glass_type_arr):
    return (
        # todo :: deal with glass types :)
        np.array([128, 196, 255]).reshape((1, 1, 3))
        * (glass_type_arr > 0)[..., np.newaxis]
    ).astype(np.uint8)


# todo :: unify functions :)
def refractive_index(qx, qy, z):
    r_squared = (qx ** 2 + qy ** 2)
    in_lens = T0 + C0 * r_squared <= z <= T0 + T1 + C1 * r_squared
    return 1 + (SOME_GLASS_RI - 1) * in_lens


# qx, qy, px, py
points = [
    (qx0, qy0, px0, py0)
    for (qx0, qy0) in [(-0.5, -0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, 0.5)]
    for (px0, py0) in [(0, 0), (0.1, 0.1)]
]

COLOURS_P = [colorsys.hsv_to_rgb(np.random.random(), 0.7, 1) for _ in range(2)]
COLOURS = [
    COLOURS_P[i % 2]
    for i in range(len(points))
]

"""
paraxial hamiltonian:
H = -(n(q, z)^2 - |p|^2) ** 0.5
hence Hamilton's equations are:
q' = dH/dp = -p / H
p' = -dH/dq = -n grad n / H
"""


def new_qq_pp(qx, qy, px, py, z, physical_eps=0.1, dz=ZS[1] - ZS[0]):
    # note: this is a terrible, non-symplectic integrator :)
    # also todo -- ideally use analytic dn/dx and dn/dy rather than numerical
    # especially since this numerical is so fragile, and we're using
    # discontinuous n...
    n = refractive_index(qx, qy, z)
    h = - (n ** 2 - px ** 2 - py ** 2) ** 0.5
    n_x_deriv = (refractive_index(qx + physical_eps, qy, z) - n) / physical_eps
    n_y_deriv = (refractive_index(qx, qy + physical_eps, z) - n) / physical_eps
    qx2 = qx - dz * px / h
    qy2 = qy - dz * py / h
    # todo :: I think there are unit failures here between dz and physical_eps
    # which makes these derivatives non-scale-invariant, aka wrong.
    px2 = px - dz * n * n_x_deriv / h
    py2 = py - dz * n * n_y_deriv / h
    return qx2, qy2, px2, py2


frames = []
for sim_idx, z in enumerate(tqdm(ZS)):
    if sim_idx % VIEW_ONE_IN_N == 0:
        # draw stuff:
        working_arr = glass_type_to_picture(glass_type_viz(z))
        for (qx, qy, px, py), draw_col in zip(points, COLOURS):
            viz_pos = (VIZ_QX - qx) ** 2 + (VIZ_QY - qy) ** 2 < VIZ_POINT_SIZE_Q ** 2
            working_arr[viz_pos, :] = np.array(draw_col) * 255
        frames.append(Image.fromarray(working_arr))
    # simulation:
    for i, (qx, qy, px, py) in enumerate(points):
        points[i] = new_qq_pp(qx, qy, px, py, z)


filename = "paraxial_hamiltonian_anim.gif"
imageio.mimsave(
    filename,
    frames,
    fps=len(frames) / ANIM_TIME_S,
)
print(f"saved {len(frames)} frames to {filename}")
