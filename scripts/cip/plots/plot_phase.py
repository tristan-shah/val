from jax import numpy as jnp
import matplotlib.pyplot as plt

from val.utils import smooth_angle_wrap

if __name__ == '__main__':

    cip = jnp.load('results/SINGLE_PENDULUM/cip/h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/traj.npy')[400:]
    ol = jnp.load('results/SINGLE_PENDULUM/ol/h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/traj.npy')[400:]

    fig, ax = plt.subplots(1, 1)
    ax.plot(smooth_angle_wrap(cip[:, 0]), cip[:, 1], label = 'CIP')
    ax.plot(smooth_angle_wrap(ol[:, 0]), ol[:, 1], label = 'OL')
    ax.legend()
    plt.show()