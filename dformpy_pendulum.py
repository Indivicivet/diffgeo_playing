import numpy as np
import matplotlib.pyplot as plt
import dformpy

SAMPLE_POINTS = 50
MAX_MOMENTUM = 12

MASS = 1
LENGTH = 1
GRAVITY = 9.8

thetatheta, pp = np.meshgrid(
    np.linspace(-np.pi, np.pi, SAMPLE_POINTS),
    np.linspace(-MAX_MOMENTUM, MAX_MOMENTUM, SAMPLE_POINTS),
)
radial_0form = dformpy.form_0(
    thetatheta,
    pp,
    (
        pp ** 2 / (2 * MASS * LENGTH)
        + MASS * GRAVITY * LENGTH * (1 - np.cos(thetatheta))
    ),
    form_0_eqn=(
        f"x**2 / {2 * MASS * LENGTH}"
        f" + {MASS * GRAVITY * LENGTH} * (1 - cos(y))"
    ),
)

plt.figure(figsize=(8, 8))
radial_0form.plot(plt.gca())
plt.show()
