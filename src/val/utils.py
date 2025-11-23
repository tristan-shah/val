import jax
from jax import Array
from jax import numpy as jnp
from mujoco.mjx import Data

def smooth_angle_wrap(theta: float):
    return jax.lax.atan2(jax.lax.sin(theta), jax.lax.cos(theta))

def split_state(xt: Array, nq: int):
    return xt[:nq], xt[nq:]

def get_state(data: Data):
    return jnp.concatenate([data.qpos, data.qvel])

def select_output(f: callable, index: int):
    return lambda *args, **kwargs: f(*args, **kwargs)[index]


def projection(u, v):
    """Projection of v onto u."""
    return (jnp.dot(v, u) / jnp.dot(u, u)) * u


def gram_schmidt(V: Array):
    """
    Fully JIT-compatible Gram–Schmidt using static loops.
    V: [d, d]
    Returns: U [d, d]
    """
    d = V.shape[1]

    # Initialize U
    U0 = jnp.zeros_like(V).at[:, 0].set(V[:, 0])

    def gs_step(U, i):
        v = V[:, i]

        # Full loop j = 0..d-1 with masking for j < i
        def proj_step(cum_proj, j):
            u_j = U[:, j]

            # mask: only accumulate projections for j < i
            mask = (j < i)
            proj = projection(u_j, v) * mask

            cum_proj = cum_proj + proj
            return cum_proj, None

        cum_proj, _ = jax.lax.scan(proj_step,
                                   jnp.zeros_like(v),
                                   jnp.arange(d))

        u_i = v - cum_proj
        U = U.at[:, i].set(u_i)
        return U, None

    U, _ = jax.lax.scan(gs_step, U0, jnp.arange(1, d))
    return U

@jax.jit
def modified_gram_schmidt(V: Array) -> Array:
    d = V.shape[1]
    U = V.at[:, 0].set(V[:, 0])

    def gs_step(i, U):
        v = V[:, i]

        def body(j, v):
            mask = (j < i)
            proj = projection(U[:, j], v) * mask
            return v - proj

        v_i = jax.lax.fori_loop(0, d, body, v)
        U = U.at[:, i].set(v_i)
        return U

    U = jax.lax.fori_loop(1, d, gs_step, U)
    return U

@jax.jit
def estimate_lyapunov_hist(A: Array, dt: float):
    """
    Computes Lyapunov exponents and returns the time history LCE(t)
    shape: (T, d)
    """
    T, d, _ = A.shape
    L = jnp.eye(d)
    LCE = jnp.zeros(d)

    def step(carry, A_t):
        L, LCE = carry

        # advance
        L = A_t @ L

        # orthogonalize
        Q = modified_gram_schmidt(L)
        norms = jnp.linalg.norm(Q, axis = 0)

        # renormalize
        L = Q / norms

        # accumulate logs
        LCE = LCE + jnp.log(norms)

        # history value for this step
        LCE_t = LCE / dt

        return (L, LCE), LCE_t

    (_, _), hist = jax.lax.scan(step, (L, LCE), A)

    # divide by time index: hist[t] /= (t+1)
    t_idx = jnp.arange(1, T+1).reshape(-1, 1)
    hist = hist / t_idx

    return hist