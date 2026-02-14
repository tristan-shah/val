# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.cip import make_compute_cip

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    # dt = 0.01
    dt = 0.05
    # horizon = 650
    horizon = 150
    shots = 512
    # steps = 1200
    steps = 600

    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    name = f'SINGLE_PENDULUM-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-dt={dt}'

    ## initialize dynamics
    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    unroll = make_unroll(step)

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
    xt = jnp.zeros(dyn.state_dim)
    # xt = xt.at[0].set(jnp.pi)

    # U = jnp.zeros((shots, horizon, dyn.control_dim))
    # cip = batch_compute_cip(xt, U)
    # print(cip.shape)
    # mpc(xt, key)



    # U = jnp.zeros((horizon, dyn.control_dim))
    # cip_stable = compute_cip(xt, U)
    # X = unroll(xt, U)

    # linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    # fx, fu = linearize(X[:-1], U)

    # from val.info import make_compute_rate, compute_volume
    # Y, V, W, K = compute_volume(fx, fu, 1.0, 1.0)




    # T = jnp.arange(1, horizon + 1)
    # ol_entropy = jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T) * dt) ## open loop
    # cl_entropy = jnp.linalg.slogdet(W).logabsdet / (2 * jnp.flip(T) * dt) ## closed loop
    # cip = ol_entropy - cl_entropy

    # fig, ax = plt.subplots(1, 1)
    # fig.suptitle(f'Single Pendulum x0 = {xt}')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('CIP (nats/s)')
    # ax.plot(T * dt, cip, label = 'CIP')
    # ax.plot(T * dt, cip_stable, label = 'CIP Log Domain')
    # ax.legend()
    # fig.savefig('cip_fixed.png', dpi = 300)
    # plt.show()







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