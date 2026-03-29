from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.cip import make_compute_cip
from val.info import make_compute_rate


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--component', type = str, default = 'ol')
    parser.add_argument('--horizon', type = int, default = 400)
    args = parser.parse_args()

    seed = args.seed
    key = jax.random.PRNGKey(seed)
    component = args.component #'ol'

    dt = 0.01
    horizon = args.horizon
    shots = 512
    steps = 1200

    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    alpha = 1.0
    rho = 0.9
    gamma = 1.0

    name = f'seed={seed}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'
    # root = Path(f'results/CART_POLE/{component}')
    # root = Path(f'results/CART_POLE/exponential_domain/{component}')
    root = Path(f'results/CART_POLE/experimental/{component}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    ## initialize dynamics
    dyn = Dynamics('xml/cart_pole.xml', dt = dt)
    step = make_step(dyn)
    unroll = make_unroll(step)

    compute_cip = make_compute_cip(dyn)
    batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes = (None, 0)))
    # batch_compute_cip = make_compute_rate(dyn, component = component)


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

    ## get initial state
    xt = jnp.zeros(dyn.state_dim)

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros((steps, 3))

    for t in range(steps):
        key, subkey = jax.random.split(key)
        # ut, J, info = mpc(xt, subkey)

        ut, J, info, U = mpc(xt, subkey)

        # fig, ax = plt.subplots(1, 1)
        # ax.plot(U[:, :, 0].T, alpha = 0.1, color = 'blue')
        # ax.set_xlabel('Planning Horizon')
        # ax.set_ylabel('Control')
        # ax.set_ylim(-1.1, 1.1)
        # fig.tight_layout()
        # fig.savefig(f'controls_{t}.png', dpi = 300)
        # plt.close(fig)





        xt = step(xt, ut)
        print(t, xt, ut, J)

        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(jnp.array([info['cip'], info['ol'], info['cl']]))

    jnp.save(path / 'hist.npy', hist)
    jnp.save(path / 'traj.npy', X)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist[:, 0], label = 'CIP')
    ax.plot(T * dt, hist[:, 1], label = 'OL')
    ax.plot(T * dt, hist[:, 2], label = 'CL')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'cip.png', dpi = 300)
    plt.show()

    dyn.render(X, path = path / 'vid.mp4', skip = 2, distance = 4)