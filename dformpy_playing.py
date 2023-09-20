import numpy as np
import matplotlib.pyplot as plt
import dformpy

SAMPLE_POINTS = 10

xx, yy = np.meshgrid(
    np.linspace(-1, 1, SAMPLE_POINTS),
    np.linspace(-1, 1, SAMPLE_POINTS),
)
radial_0form = dformpy.form_0(
    xx,
    yy,
    (xx ** 2 + yy ** 2) ** 0.5,
    form_0_eqn="(x**2 + y ** 2) ** 0.5",
)

plt.figure(figsize=(8, 8))
radial_0form.ext_d().plot(plt.gca())
plt.show()
