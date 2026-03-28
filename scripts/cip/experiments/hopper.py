import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.cem import CEM
from val.cip import make_compute_cip

def hopper_initial_state():
    return jnp.array([0.0, -0.245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    integrator = 'rk4'
    component = 'ol'
    dt = 0.01
    horizon = 512
    shots = 512
    steps = 1000
    iterations = 5
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9
    gear = 25
    
    root = Path(f'results/CIP/UNRESTRICTED_HOPPER/{component}')
    name = f'integrator={integrator}-gear={gear}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    ## initialize dynamics
    dyn = Dynamics('xml/unrestricted_hopper.xml', dt = dt, integrator = integrator)

    ## override default gear strength
    dyn.mjx_model = dyn.mjx_model.replace(
        actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(gear)
    )
    
    step = make_step(dyn)
    compute_cip = make_compute_cip(dyn, component)
    ## vectorize over batches of trajectories
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
    
    ## get initial state
    xt = hopper_initial_state()

    # '''
    # test passive dynamics
    # '''
    # from val import make_unroll
    # unroll = make_unroll(step)
    # traj_linerize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    # U = jnp.zeros((horizon, dyn.control_dim))
    # # U = jax.random.uniform(key, (horizon, dyn.control_dim)) * 2.0 - 1.0
    # X = unroll(xt, U)

    # fx, fu = traj_linerize(X[:-1], U)
    
    # horizon, dx, dx = fx.shape

    # Q = jnp.eye(dx)
    # Y_inv_t = Q.copy()

    # reward = jnp.zeros(horizon)

    # for t in reversed(range(horizon)):
    #     # r = jnp.linalg.slogdet(Y_inv_t + fx[t] @ fx[t].T).logabsdet
    #     r = jnp.linalg.slogdet(Q + fx[t] @ fx[t].T).logabsdet
    #     Y_inv_t = Q - fx[t].T @ jnp.linalg.inv(Y_inv_t + fx[t] @ fx[t].T) @ fx[t]
    #     print(r)

    #     reward = reward.at[t].set(r)
        
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(reward)
    # ax.set_title('Reward During Backwards Recursion')
    # ax.set_ylabel('Instantanious Reward')
    # ax.set_xlabel('Timestep (s)')
    # fig.savefig(f'{integrator}_hopper_reward_{dt}.png', dpi = 300)
    # plt.show()

    # dyn.render(X, path = 'hopper.mp4', skip = 1, distance = 4)

    '''
    run mpc
    '''
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros(steps)

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J, info, U = mpc(xt, subkey)
        xt = step(xt, ut)
        print(t, xt, ut, J)

        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(J)

    jnp.save(path / 'hist.npy', hist)
    jnp.save(path / 'traj.npy', X)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(path / 'cip.png', dpi = 300)
    plt.show()

    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 4)