import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'

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

    dt = 0.01
    horizon = 512
    shots = 512
    steps = 500
    iterations = 10
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.999

    gear = 100
    name = f'HOPPER-gear={gear}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'

    ## initialize dynamics
    dyn = Dynamics('xml/hopper.xml')
    
    step = make_step(dyn)

    compute_cip = make_compute_cip(dyn)
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
        smoothing)
    
    ## get initial state
    xt = hopper_initial_state()

    # '''
    # test passive dynamics
    # '''
    # xt = jnp.array([-0.73826802, -1.19743042, -4.70699487, -1.44451927, -2.15036256,  0.79326958, 0.03450733,  0.02324211,  0.02154036,  3.28547185, -2.03091685, -0.16750272])
    # from val import make_unroll
    # unroll = make_unroll(step)
    # # U = jnp.zeros((horizon, dyn.control_dim))
    # U = jax.random.uniform(key, (horizon, dyn.control_dim)) * 2.0 - 1.0
    # X = unroll(xt, U)
    # dyn.render(X, path = 'hopper.mp4', skip = 1)


    '''
    run mpc
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