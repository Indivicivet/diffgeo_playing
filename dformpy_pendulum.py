import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
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
d_hamiltonian = hamiltonian.ext_d()

hamiltonian_flow = d_hamiltonian.contravariant(
    g=np.array([[0, 1], [-1, 0]]).reshape(2, 2, 1)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
hamiltonian.plot(ax1)
hamiltonian_flow.plot(ax2)
plt.show()

# unreachable wip code:
particle_pt = np.array([0, 4], dtype=float)
point_plot = ax1.scatter(*particle_pt)


def update_point(frame):
    global particle_pt
    # todo :: unclear if anything sensible to do here with dformpy :(
    # seems like symbolic entries are not trivially eval-able
    print(d_hamiltonian.F_x)
    particle_pt += [
        d_hamiltonian.F_x * particle_pt[0],
        d_hamiltonian.F_y * particle_pt[1],
    ]
    point_plot.set_offsets(particle_pt)


_anim = mpl_anim.FuncAnimation(fig, update_point, frames=999, interval=100)
plt.show()
