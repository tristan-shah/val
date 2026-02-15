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
    name = f'UNRESTRICTED_HOPPER-gear={gear}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'

    ## initialize dynamics
    dyn = Dynamics('xml/unrestricted_hopper.xml', dt = 0.01)
    
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
        smoothing,
        rho)
    
    ## get initial state
    xt = hopper_initial_state()

    # # '''
    # # test passive dynamics
    # # '''
    # # # xt = jnp.array([-0.95117254, -0.77809011, -8.05780964, -2.1339089, -1.9284697, 0.68816111, -1.61971951, 0.58293853, 2.61751397, 6.11707583, -2.92111298, 2.75567617])
    # # # xt = jnp.array([-0.06082055, -1.11377958, -4.49441237, -1.2071687, -2.59668601, -0.81942782, 0.11672043, 0.36908223, 0.9374153, 5.4942232, -2.63953188, -0.13625338])
    # # # xt = jnp.array([-0.08715446, -1.18272198, -4.66715652, -1.71564075, -2.17217936, -0.81248989, 0.06693304, 0.26217543, 0.55690369, -0.36191009, 0.67620763, -0.02986369])
    # # xt = jnp.array([-0.84561067, -0.77951274, -8.43766739, -2.59336965, -1.63642628, 0.77351244, 1.20768491, -1.31422078, 3.59244821, 1.05036794, 0.25376109, -0.07993622])

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