import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.info import make_compute_rate

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    dt = 0.01
    horizon = 256
    shots = 128
    steps = 3000

    iterations = 1
    elite_frac = 0.1
    keep_frac = 0.3
    smoothing = 0.1
    alpha = 1.0
    rho = 0.9
    gamma = 1.0

    name = f'ANT-h={horizon}-gamma={gamma}-shots={shots}-iter={iterations}-elite={elite_frac}_keep={keep_frac}-smooth={smoothing}-alpha={alpha}-dt={dt}'

    dyn = Dynamics('xml/ant.xml', dt = 0.01)
    step = make_step(dyn)
    print(dyn.state_dim, dyn.control_dim)

    mpc = CEM(
        dyn, 
        make_compute_rate(dyn, alpha, gamma),
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        keep_frac,
        smoothing,
        rho)
    
    ## initial state of ant
    xt = jnp.array([
        4.63033074e-09, -1.33166722e-08,  5.88738445e-01,  1.00000000e+00,
        4.03076686e-10,  1.40153441e-10,  2.18994597e-19,  1.34050243e-09,
        1.08063526e+00, -6.48803570e-10, -1.08063526e+00, -1.34050247e-09,
        -1.08063526e+00,  6.48803545e-10,  1.08063526e+00,  1.69124988e-09,
        -4.86398140e-09, -2.83866190e-03,  7.92753681e-09,  2.75647608e-09,
        -6.05048348e-18,  4.30372771e-10, -1.06604418e-02, -2.08300861e-10,
        1.06604411e-02, -4.30372777e-10,  1.06604430e-02,  2.08300867e-10,
        -1.06604437e-02])

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros(steps)

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J = mpc(xt, subkey)
        xt = step(xt, ut)
        print(t, xt, ut, J)

        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(J)

    jnp.save(name + '-hist.npy', hist)
    jnp.save(name + '-traj.npy', X)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(name + '.png', dpi = 300)
    plt.show()

    dyn.render(X, path = name + '.mp4', skip = 1, distance = 4)

