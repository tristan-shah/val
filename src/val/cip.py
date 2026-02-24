import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum

from val import Dynamics, make_step, make_unroll

@jax.jit
def compute_entropy(fx: Array, fu: Array):

    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du)

    def scan_fn(carry: tuple, inputs: tuple[Array, Array]):
        ## logdet_Y_t: log det of Y_t 
        ## logdet_W_t: log det of W_t
        ## Y_inv_t: inverse of Y_t 
        ## W_inv_t: inverse of W_t
        ## V_t: solution to Riccati Equation
        logdet_Y_t, logdet_W_t, Y_inv_t, W_inv_t, V_t = carry
        fx_t, fu_t = inputs

        S_inv = jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t)
        ## optimal feedback gain
        K_t = - S_inv @ fu_t.T @ V_t @ fx_t
        ## closed loop entropy
        D_t = fx_t + fu_t @ K_t

        ## propagate in log space
        logdet_Y_t = logdet_Y_t + jnp.linalg.slogdet(Y_inv_t + fx_t @ fx_t.T).logabsdet
        logdet_W_t = logdet_W_t + jnp.linalg.slogdet(W_inv_t + D_t @ D_t.T).logabsdet

        ## propagate inverses
        Y_inv_t = Q - fx_t.T @ jnp.linalg.inv(Y_inv_t + fx_t @ fx_t.T) @ fx_t
        W_inv_t = Q - D_t.T @ jnp.linalg.inv(W_inv_t + D_t @ D_t.T) @ D_t
        ## propagate Riccati
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t

        carry = (logdet_Y_t, logdet_W_t, Y_inv_t, W_inv_t, V_t)
        return carry, (logdet_Y_t, logdet_W_t)

    logdet_Y_T = jnp.array(0.0)
    logdet_W_T = jnp.array(0.0)
    Y_inv_T = Q
    W_inv_T = Q
    V_T = Q

    init = (logdet_Y_T, logdet_W_T, Y_inv_T, W_inv_T, V_T)
    _, (logdet_Y, logdet_W) = jax.lax.scan(scan_fn, init = init, xs = (fx, fu), reverse = True)
    return logdet_Y, logdet_W

def make_compute_cip(dyn: Dynamics):

    step = make_step(dyn)
    traj_linerize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    unroll = make_unroll(step)
    dt = dyn.mjx_model.opt.timestep

    def compute_cip(xt: Array, U: Array):
        X = unroll(xt, U)
        fx, fu = traj_linerize(X[:-1], U)

        ## calculate logdet of the recursions
        logdet_Y, logdet_W = compute_entropy(fx, fu)

        horizon = fx.shape[0]
        T = jnp.arange(1, horizon + 1)

        ## normalize by time
        ol = logdet_Y / (2 * jnp.flip(T) * dt)
        cl = logdet_W / (2 * jnp.flip(T) * dt)
        cip = ol - cl
        return cip[0]
    
    return jax.jit(compute_cip)

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

@jax.jit
def compute_entropy_approximation(fx: Array, fu: Array):
    K = compute_feedback(fx, fu)
    D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')

    ol_matrix = einsum(fx, fx, 't x1 x, t x2 x -> t x1 x2')
    cl_matrix = einsum(D, D, 't x1 x, t x2 x -> t x1 x2')

    dx = fx.shape[-1]
    I = jnp.eye(dx)

    ol = jnp.linalg.slogdet(I + ol_matrix).logabsdet
    cl = jnp.linalg.slogdet(I + cl_matrix).logabsdet

    return ol, cl

def make_compute_cip_approximation(dyn: Dynamics):

    step = make_step(dyn)
    traj_linerize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    unroll = make_unroll(step)
    dt = dyn.mjx_model.opt.timestep

    def compute_cip_approximation(xt: Array, U: Array):
        X = unroll(xt, U)
        fx, fu = traj_linerize(X[:-1], U)
        ol, cl = compute_entropy_approximation(fx, fu)
        cip = ol - cl
        return jnp.mean(cip) / (2 * dt)
    
    return jax.jit(compute_cip_approximation)

if __name__ == '__main__':

    dt = 0.05
    horizon = 100

    ## initialize dynamics
    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    
    step = make_step(dyn)
    linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    unroll = make_unroll(step)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(jnp.pi)

    U = jnp.zeros((horizon, dyn.control_dim))

    compute_cip = make_compute_cip(dyn)
    cip = compute_cip(xt, U)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'Single Pendulum x0 = {xt}')
    ax.plot(cip)
    # ax.plot(ol, label = 'Open Loop')
    # ax.plot(cl, label = 'Closed Loop')
    # ax.plot(Z, label = 'Stable Open Loop', linestyle = 'dashed', color = 'red')
    # ax.plot(X)
    ax.legend()
    # fig.savefig('fixed.png', dpi = 300)
    plt.show()