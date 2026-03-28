from argparse import ArgumentParser
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
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
    component = args.component

    dt = 0.01
    horizon = args.horizon
    shots = 512
    steps = 2000 #1200

    iterations = 10
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9

    name = f'seed={seed}-gear=6.0-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'
    # root = Path(f'results/DOUBLE_PENDULUM/{component}')
    root = Path(f'results/DOUBLE_PENDULUM/exponential_domain/{component}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    dyn = Dynamics('xml/double_pendulum.xml', dt = dt, integrator = 'implicitfast')
    step = make_step(dyn)
    print(dyn.state_dim, dyn.control_dim)

    # compute_cip = make_compute_cip(dyn, component)
    # batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes = (None, 0)))
    batch_compute_cip = make_compute_rate(dyn, component = component)

    mpc = CEM(
        dyn,
        batch_compute_cip,
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        smoothing,
        rho)
    
    ## initial state
    xt = jnp.zeros(dyn.state_dim)

    from val.dynamics import make_unroll

    horizon = 600

    xt = xt.at[0].set(2.14)
    unroll = make_unroll(step)
    traj_linerize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    U = jnp.zeros((horizon, dyn.control_dim))
    X = unroll(xt, U)

    fx, fu = traj_linerize(X[:-1], U)
    
    horizon, dx, dx = fx.shape

    Q = jnp.eye(dx)
    Y_inv_t = Q.copy()

    reward = jnp.zeros(horizon)

    for t in reversed(range(horizon)):
        r = jnp.linalg.slogdet(Y_inv_t + fx[t] @ fx[t].T + Q * 1e-3).logabsdet
        Y_inv_t = Q - fx[t].T @ jnp.linalg.inv(Y_inv_t + fx[t] @ fx[t].T) @ fx[t]
        print(r)

        reward = reward.at[t].set(r)
        
    fig, ax = plt.subplots(1, 1)
    ax.plot(reward)
    ax.set_title('Reward During Backwards Recursion')
    ax.set_ylabel('Instantanious Reward')
    ax.set_xlabel('Timestep (s)')
    fig.savefig('double_pendulum_reward.png', dpi = 300)
    plt.show()

    dyn.render(X, path = 'double_pendulum.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))




    # X = jnp.zeros((steps + 1, dyn.state_dim))
    # X = X.at[0].set(xt)

    # hist = jnp.zeros((steps, 3))

    # for t in range(steps):
    #     key, subkey = jax.random.split(key)
    #     ut, J, info, U = mpc(xt, subkey)

    #     xt = step(xt, ut)
    #     print(t, xt, ut, J)

    #     X = X.at[t+1].set(xt)
    #     # hist = hist.at[t].set(J)
    #     hist = hist.at[t].set(jnp.array([info['cip'], info['ol'], info['cl']]))

    # jnp.save(path / 'hist.npy', hist)
    # jnp.save(path / 'traj.npy', X)

    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('nats / s')
    # T = jnp.arange(0, steps)
    # ax.plot(T * dt, hist[:, 0], label = 'CIP')
    # ax.plot(T * dt, hist[:, 1], label = 'OL')
    # ax.plot(T * dt, hist[:, 2], label = 'CL')
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(path / 'cip.png', dpi = 300)
    # plt.show()

    # dyn.render(X, path = path / 'vid.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))