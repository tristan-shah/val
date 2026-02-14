# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

import colorednoise

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.cip import make_compute_cip



from jax import Array
from einops import einsum

@jax.jit
def compute_feedback(fx: Array, fu: Array):

    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du)

    def scan_fn(V_t: Array, inputs: tuple[Array, Array]):
        
        fx_t, fu_t = inputs

        S_inv = jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t)
        ## optimal feedback gain
        K_t = - S_inv @ fu_t.T @ V_t @ fx_t

        ## propagate Riccati
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t

        return V_t, K_t

    V_T = Q
    _, K = jax.lax.scan(scan_fn, init = V_T, xs = (fx, fu), reverse = True)
    return K

# @jax.jit
def compute_entropy_approximation(fx: Array, fu: Array):
    K = compute_feedback(fx, fu)
    D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')

    ol_matrix = einsum(fx, fx, 't x x1, t x x2 -> t x1 x2')
    cl_matrix = einsum(D, D, 't x x1, t x x2 -> t x1 x2')

    ol_eigs = jnp.linalg.eigvalsh(ol_matrix)
    cl_eigs = jnp.linalg.eigvalsh(cl_matrix)
    
    ## softplus
    ol = jax.nn.softplus(jnp.log(ol_eigs)).sum(axis = -1)
    cl = jax.nn.softplus(jnp.log(cl_eigs)).sum(axis = -1)

    ## simple norm doesnt seem to work
    # ol = jnp.linalg.matrix_norm(fx) ** 2
    # cl = jnp.linalg.matrix_norm(D) ** 2
    return ol, cl

def make_compute_cip_approximation(dyn: Dynamics):

    step = make_step(dyn)
    traj_linerize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    unroll = make_unroll(step)

    def compute_cip_approximation(xt: Array, U: Array):
        X = unroll(xt, U)
        fx, fu = traj_linerize(X[:-1], U)

        ol, cl = compute_entropy_approximation(fx, fu)
        cip = ol - cl
        return jnp.sum(cip)
    
    return jax.jit(compute_cip_approximation)

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    dt = 0.05
    horizon = 200
    shots = 512
    steps = 600
    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9
    
    name = f'SINGLE_PENDULUM-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'

    ## initialize dynamics
    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    unroll = make_unroll(step)

    compute_cip = make_compute_cip(dyn)
    # compute_cip = make_compute_cip_approximation(dyn)
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
    xt = jnp.zeros(dyn.state_dim)

    '''
    Plot CIP
    '''
    # xt = xt.at[0].set(0.1)

    # # U = jnp.zeros((shots, horizon, dyn.control_dim))
    # # cip = batch_compute_cip(xt, U)

    # U = jnp.zeros((horizon, dyn.control_dim))
    # # cip_stable = compute_cip(xt, U)
    # X = unroll(xt, U)

    # linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    # fx, fu = linearize(X[:-1], U)
    # compute_entropy_approximation(fx, fu)

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


    '''
    Run MPC
    '''
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros(steps)

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J = mpc(xt, subkey)


        # fig, ax = plt.subplots(1, 1)
        # ax.plot(mpc.mean)
        # plt.show()
        # plt.close(fig)


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