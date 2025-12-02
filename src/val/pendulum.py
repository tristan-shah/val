import jax
from jax import Array
from jax import numpy as jnp

from val.utils import smooth_angle_wrap

@jax.jit
def init_pendulum_state(key):
    xt = jax.random.uniform(key, shape = 2) * 2 - 1
    xt = xt * jnp.array([jnp.pi, 1.0])
    return xt

@jax.jit
def normalize_pendulum_state(xt: Array, max_vel: float = 3 * jnp.pi):
    '''
    Takes the raw mujoco state and normalizes angles and velocities.
    '''
    return jnp.stack([jnp.cos(xt[0]), jnp.sin(xt[0]), xt[1] / max_vel])

@jax.jit
def pendulum_reward(xt: Array, ut: Array):
    theta = xt[0]
    theta_dot = xt[1]
    angle_cost = smooth_angle_wrap(theta - jnp.pi) ** 2
    vel_cost = 0.1 * theta_dot ** 2
    act_cost = 0.5 * ut ** 2
    return -(angle_cost + vel_cost + act_cost)

@jax.jit
def pendulum_cost(_x: Array, _u: Array):
    return -pendulum_reward(_x, _u).squeeze()