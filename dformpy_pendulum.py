import numpy as np
import matplotlib.pyplot as plt
import dformpy

SAMPLE_POINTS = 40
MAX_MOMENTUM = 20

MASS = 1
LENGTH = 1
GRAVITY = 9.8

thetatheta, pp = np.meshgrid(
    np.linspace(-np.pi, np.pi, SAMPLE_POINTS),
    np.linspace(-MAX_MOMENTUM, MAX_MOMENTUM, SAMPLE_POINTS),
)
hamiltonian = dformpy.form_0(
    thetatheta,
    pp,
    (
        pp ** 2 / (2 * MASS * LENGTH)
        + MASS * GRAVITY * LENGTH * (1 - np.cos(thetatheta))
    ),
    form_0_eqn=(
        f"y**2 / {2 * MASS * LENGTH}"
        f" + {MASS * GRAVITY * LENGTH} * (1 - cos(x))"
    ),
)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
hamiltonian.plot(ax1)
hamiltonian.ext_d().contravariant(
    g=np.array([[0, 1], [-1, 0]]).reshape(2, 2, 1)
).plot(ax2)
plt.show()
