import jax
from jax import Array
from jax import numpy as jnp

from val import Dynamics, make_unroll

@jax.jit
def compute_vol(fx: Array, fu: Array):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du) * 1.0
    gamma = 1.0

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t = carry
        fx_t, fu_t = inputs

        Y_t = Q + fx_t.T @ Y_t @ fx_t
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t) @ fu_t.T @ V_t @ fx_t
        # Y_t = Q + gamma * fx_t.T @ Y_t @ fx_t
        # V_t = Q + gamma * fx_t.T @ V_t @ fx_t - gamma ** 2 * fx_t.T @ V_t @ fu_t @ jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t) @ fu_t.T @ V_t @ fx_t

        carry = (Y_t, V_t)

        return carry, carry
    
    _, (Y, V) = jax.lax.scan(scan_fn, init = (Q, Q), xs = (fx, fu), reverse = True)
    return Y, V

def make_compute_entropy(step: callable):
    
    traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step)

    def compute_entropy(x0: Array, U: Array):

        X = unroll(x0, U)
        fx, fu = traj_linerize(X[:-1], U)
        Y, V = compute_vol(fx, fu)

        ol_entropy = jnp.linalg.slogdet(Y).logabsdet
        cl_entropy = jnp.linalg.slogdet(V).logabsdet

        return ol_entropy, cl_entropy
    
    return jax.jit(compute_entropy)