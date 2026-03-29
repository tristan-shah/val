from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    # root = Path(f'results/SINGLE_PENDULUM/{component}')
    # root = Path(f'results/SINGLE_PENDULUM/exponential_domain/{component}')
    # root = Path(f'results/SINGLE_PENDULUM/qpos/{component}')
    root = Path(f'results/SINGLE_PENDULUM/experimental/{component}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    ## initialize dynamics
    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    unroll = make_unroll(step)

    compute_cip = make_compute_cip(dyn, component)
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


    # '''
    # plotting
    # '''
    # horizon = 600
    # # xt = xt.at[0].set(3.14)
    # unroll = make_unroll(step)
    # traj_linerize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    # U = jnp.zeros((horizon, dyn.control_dim))
    # X = unroll(xt, U)
    # fx, fu = traj_linerize(X[:-1], U)

    # from val.cip import make_compute_entropy

    # compute_entropy = make_compute_entropy(dyn.nq)

    # ol, cl = compute_entropy(fx, fu)

    
    # horizon, dx, dx = fx.shape

    # Q = jnp.eye(dx)
    # Y_inv_t = Q.copy()

    # reward = jnp.zeros(horizon)

    # for t in reversed(range(horizon)):
    #     M = Y_inv_t + fx[t] @ fx[t].T + Q * 1e-6
    #     r = jnp.linalg.slogdet(M[:dyn.nq, :dyn.nq]).logabsdet
    #     Y_inv_t = Q - fx[t].T @ jnp.linalg.inv(Y_inv_t + fx[t] @ fx[t].T) @ fx[t]
    #     print(r)

    #     reward = reward.at[t].set(r)
        
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(reward)
    # ax.plot(ol)
    # ax.set_title('Reward During Backwards Recursion')
    # ax.set_ylabel('Instantanious Reward')
    # ax.set_xlabel('Timestep (s)')
    # fig.savefig('single_pendulum_reward.png', dpi = 300)
    # plt.show()

    # dyn.render(X, path = 'vid.mp4', skip = 1, distance = 4)



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