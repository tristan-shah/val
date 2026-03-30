from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.cip import make_compute_cip
from val.info import make_compute_rate


# U = jnp.zeros((horizon, dyn.control_dim))
# # U = jax.random.normal(jax.random.key(0), (horizon, dyn.control_dim))
# unroll = make_unroll(step)
# X = unroll(xt, U)
# dyn.render(X, path = 'triple_pendulum.mp4', skip = 2, distance = 5)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--component', type = str, default = 'ol')
    parser.add_argument('--horizon', type = int, default = 150)
    args = parser.parse_args()

    seed = args.seed
    key = jax.random.PRNGKey(seed)

    component = args.component

    dt = 0.05
    horizon = args.horizon
    shots = 512
    steps = 600
    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9

    name = f'seed={seed}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'
    root = Path(f'results/TRIPLE_PENDULUM/{component}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    dyn = Dynamics('xml/triple_pendulum.xml', dt = dt)
    step = make_step(dyn)

    compute_cip = make_compute_cip(dyn, component)
    batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes = (None, 0)))

    ## initialize agent
    mpc = CEM(
        dyn,
        batch_compute_cip,
        shots,
        horizon, 
        iterations, 
        elite_frac,
        smoothing,
        rho)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(jnp.pi) ## start from bottom

    '''
    Run MPC
    '''
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros((steps, 3))
    controls = jnp.zeros((steps, dyn.control_dim))

    for t in range(steps):

        key, subkey = jax.random.split(key)
        ut, J, info, U = mpc(xt, subkey)

        xt = step(xt, ut)
        print(t, xt, ut, J)

        controls = controls.at[t].set(ut)
        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(jnp.array([info['cip'], info['ol'], info['cl']]))

    jnp.save(path / 'hist.npy', hist)
    jnp.save(path / 'traj.npy', X)
    jnp.save(path / 'U.npy', controls)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist[:, 0], label = 'CIP')
    ax.plot(T * dt, hist[:, 1], label = 'OL')
    ax.plot(T * dt, hist[:, 2], label = 'CL')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'metrics.png', dpi = 300)
    plt.show()

    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 4)