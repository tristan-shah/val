# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'
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
    seed = 4
    key = jax.random.PRNGKey(seed)

    component = 'cip'

    dt = 0.05
    horizon = 150
    shots = 512
    steps = 600
    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9
    
    name = f'seed={seed}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'
    # root = Path(f'results/SINGLE_PENDULUM/{component}')
    root = Path(f'results/SINGLE_PENDULUM/exponential_domain/{component}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    ## initialize dynamics
    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    unroll = make_unroll(step)

    # compute_cip = make_compute_cip(dyn, component)
    # ## vectorize over batches of trajectories
    # batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes = (None, 0)))

    batch_compute_cip = make_compute_rate(dyn, component = component)

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

    '''
    Plot CIP
    '''
    # xt = xt.at[0].set(3.14)

    # U = jnp.zeros((horizon, dyn.control_dim))
    # X = unroll(xt, U)

    # linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    # fx, fu = linearize(X[:-1], U)
    # compute_entropy_approximation(fx, fu)

    '''
    Run MPC
    '''
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