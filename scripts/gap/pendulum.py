# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
from jax import Array
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.info import make_compute_rate
from val.info import compute_volume
from val.utils import estimate_lyapunov_hist

# def make_compute_rate(dyn: Dynamics, alpha: float = 1.0, gamma: float = 1.0):

#     ## helper functions
#     step = make_step(dyn)
#     traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
#     batch_traj_linearize = jax.jit(jax.vmap(traj_linerize))
#     unroll = make_unroll(step)
#     batch_unroll = jax.jit(jax.vmap(unroll, in_axes = (None, 0)))
#     batch_compute_volume = jax.jit(jax.vmap(compute_volume, in_axes = (0, 0, None, None)))

#     dt = dyn.mjx_model.opt.timestep

#     batch_estimate_lyapunov_hist = jax.vmap(estimate_lyapunov_hist, in_axes = (0, None))

#     def compute_rate(xt: Array, U_batch: Array):

#         horizon = U_batch.shape[1]
#         T = jnp.arange(1, horizon + 1)

#         ## unroll trajectories
#         X_batch = batch_unroll(xt, U_batch)
#         ## linearize the batch of trajectories
#         fx_batch, fu_batch = batch_traj_linearize(X_batch[:, :-1, :], U_batch)
#         # ## compute entropy of each trajectory (manually setting alpha = 1.0)
#         Y, V, W, K = batch_compute_volume(fx_batch, fu_batch, alpha, gamma)

#         # D = fx_batch + einsum(fu_batch, K, 'b t x1 u, b t u x2 -> b t x1 x2')
        
#         # ol_entropy = jnp.flip(
#         #     batch_estimate_lyapunov_hist(fx_batch, dt).clip(min = 0.0).sum(axis = 2),
#         #     axis = 1)
        
#         # cl_entropy = jnp.flip(
#         #     batch_estimate_lyapunov_hist(D, dt).clip(min = 0.0).sum(axis = 2),
#         #     axis = 1)

#         # fig, ax = plt.subplots(1, 2)
#         # for i in range(512):
#         #     ax[0].plot(ol_entropy[i, :])
#         #     ax[1].plot(cl_entropy[i, :])
#         # plt.show()

#         ## compute entropy
#         ol_entropy = jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T) * dt)
#         cl_entropy = jnp.linalg.slogdet(W).logabsdet / (2 * jnp.flip(T) * dt)


#         # fig, ax = plt.subplots(1, 2)
#         # for i in range(512):
#         #     ax[0].plot(ol_entropy[i, :])
#         #     ax[1].plot(cl_entropy[i, :])
#         # plt.show()

#         # exit()

#         ## convert entropy into information
#         information = ol_entropy - cl_entropy

#         rate = information

#         return rate[:, 0]
    
#     return jax.jit(compute_rate)

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    dt = 0.01 #0.05
    horizon = 600 #200
    shots = 512
    steps = 1200

    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    alpha = 1.0
    rho = 0.9
    gamma = 1.0

    name = f'SINGLE_PENDULUM-h={horizon}-gamma={gamma}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-alpha={alpha}-dt={dt}'

    ## initialize dynamics
    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    unroll = make_unroll(step)

    ## initialize agent
    mpc = CEM(
        dyn,
        make_compute_rate(dyn, alpha, gamma),
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        smoothing,
        rho)
    
    ## get initial state
    xt = jnp.zeros(dyn.state_dim)
    # xt = xt.at[0].set(jnp.pi)

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