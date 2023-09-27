import numpy as np
from PIL import Image
import imageio
import colorsys


VIZ_SAMPLES = 512
Z_SAMPLES = 100
ANIM_TIME_S = 5
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

T0 = 3  # air thick
T1 = 3  # glass thick
C0 = 1  # curvature coeff front face
C1 = -1  # curvature coeff back face

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
def glass_type_pt(qx, qy, z):
    r_squared = (qx ** 2 + qy ** 2)
    return T0 + C0 * r_squared <= z <= T0 + T1 + C1 * r_squared


def glass_to_ri(glass_type):
    return 1 + (SOME_GLASS_RI - 1) * glass_type


# qx, qy, px, py
points = [
    (-0.5, -0.5, 0, 0),
    (-0.5, -0.5, 0.1, 0.1),
    (0.5, 0.5, 0, 0),
    (0.5, 0.5, 0.1, 0.1),
]

COLOURS = [
    colorsys.hsv_to_rgb(np.random.random(), 0.7, 1)
    for _ in range(len(points))
]

"""
paraxial hamiltonian:
H = -(n(q, z)^2 - |p|^2) ** 0.5
hence Hamilton's equations are:
q' = dH/dp = -p / H
p' = dH/dq = (q' . grad n) / H
"""


def new_qq_pp(qx, qy, px, py, z):
    h = - (glass_to_ri(glass_type_pt(qx, qy, z)) ** 2 - px ** 2 - py ** 2) ** 0.5
    qx2 = qx - px / h
    qy2 = qy - py / h
    px2 = px
    py2 = py
    return qx2, qy2, px2, py2


frames = []
for z in ZS:
    working_arr = glass_type_to_picture(glass_type_viz(z))
    for i, ((qx, qy, px, py), draw_col) in enumerate(zip(points, COLOURS)):
        viz_pos = (VIZ_QX - qx) ** 2 + (VIZ_QY - qy) ** 2 < VIZ_POINT_SIZE_Q ** 2
        working_arr[viz_pos, :] = np.array(draw_col) * 255
        points[i] = new_qq_pp(qx, qy, px, py, z)
    frames.append(Image.fromarray(working_arr))

filename = "paraxial_hamiltonian_anim.gif"
imageio.mimsave(
    filename,
    frames,
    fps=len(frames) / ANIM_TIME_S,
)
print(f"saved {len(frames)} frames to {filename}")
