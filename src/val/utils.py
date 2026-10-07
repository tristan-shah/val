import jax
from jax import Array
from jax import numpy as jnp
from mujoco.mjx import Data


def split_state(xt: Array, nq: int):
    '''
    Splits a flat state vector into (qpos, qvel).
    '''
    return xt[:nq], xt[nq:]


def get_state(data: Data):
    '''
    Flattens an mjx data object into a state vector [qpos, qvel].
    '''
    return jnp.concatenate([data.qpos, data.qvel])


def ar1_noise(key, shots: int, horizon: int, control_dim: int, rho: float = 0.9):
    '''
    Returns noise of shape (shots, horizon, control_dim) with AR(1) time correlation
    (the colored-noise action sampling of iCEM, Pinneri et al. 2021).
    '''
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (horizon, shots, control_dim))

    alpha = jnp.sqrt(1.0 - rho ** 2)

    def step(prev, curr):
        out = rho * prev + alpha * curr
        return out, out

    init = eps[0]                                   ## (shots, control_dim)
    _, ys = jax.lax.scan(step, init, eps[1:])

    noise = jnp.concatenate([init[None, ...], ys], axis = 0)    ## (horizon, shots, control_dim)

    return jnp.transpose(noise, (1, 0, 2))
