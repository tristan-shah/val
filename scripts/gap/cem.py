import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

@jax.jit
def compute_volume(fx: Array, fu: Array, alpha: float):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du) * alpha

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t, W_t = carry
        fx_t, fu_t = inputs

        S_inv = jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t)
        ## feedback gain
        K_t = - S_inv @ fu_t.T @ V_t @ fx_t
        ## open loop entropy
        Y_t = Q + fx_t.T @ Y_t @ fx_t
        ## riccati equation
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t
        ## closed loop entropy
        W_t = Q + (fx_t + fu_t @ K_t).T @ W_t @ (fx_t + fu_t @ K_t)

        carry = (Y_t, V_t, W_t)

        return carry, (Y_t, V_t, W_t, K_t)
    
    _, (Y, V, W, K) = jax.lax.scan(scan_fn, init = (Q, Q, Q), xs = (fx, fu), reverse = True)
    return Y, V, W, K

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    dt = 0.05
    horizon = 50
    shots = 256
    eps = 0.1
    iterations = 50
    alpha = 1.0

    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]

    ## helper functions
    step = make_step(dyn)
    traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    batch_traj_linearize = jax.jit(jax.vmap(traj_linerize))
    unroll = make_unroll(step)
    batch_unroll = jax.jit(jax.vmap(unroll, in_axes = (None, 0)))
    batch_compute_volume = jax.jit(jax.vmap(compute_volume, in_axes = (0, 0, None)))


    theta = 0.0
    x0 = jnp.zeros(dyn.state_dim)
    xt = x0.copy()

    U = jnp.zeros((horizon, dyn.control_dim))
    key, subkey = jax.random.split(key)
    U_batch = U[None, :, :] + (jax.random.normal(subkey, (shots, horizon, dyn.control_dim)) * eps)
    U_batch = U_batch.clip(low[None, None, :], high[None, None, :])

    ## unroll trajectories
    X_batch = batch_unroll(xt, U_batch)
    ## linearize the batch of trajectories
    fx_batch, fu_batch = batch_traj_linearize(X_batch[:, :-1, :], U_batch)
    ## compute entropy of each trajectory
    Y, V, W, K_batch = batch_compute_volume(fx_batch, fu_batch, alpha)

    ## compute entropy
    ol_entropy = jnp.linalg.slogdet(Y).logabsdet
    cl_entropy = jnp.linalg.slogdet(V).logabsdet
    ## convert entropy into information
    information = ol_entropy - cl_entropy

    T = jnp.arange(1, horizon + 1)
    rate = information / (jnp.flip(T) * dt)

    J = rate[:, 0]
    print(J)

    # fig, ax = plt.subplots(1, 1)
    # for i in range(shots):
    #     ax.plot(U_batch[i])
    # plt.show()