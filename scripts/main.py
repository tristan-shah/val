import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from val import Dynamics

@jax.jit
def normalize_pendulum_state(xt: Array, max_vel: float = 3 * jnp.pi):
    '''
    Takes the raw mujoco state and normalizes angles and velocities.
    '''
    return jnp.stack([jnp.cos(xt[0]), jnp.sin(xt[0]), xt[1] / max_vel])

def build_pendulum_critic(h_dim: int, rngs: nnx.Rngs):

    critic = nnx.Sequential(
        nnx.Linear(4, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, 1, rngs = rngs)
    )

    return critic

def build_pendulum_policy(h_dim: int, rngs: nnx.Rngs):

    policy = nnx.Sequential(
        nnx.Linear(3, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, h_dim, rngs = rngs),
        nnx.gelu,
        nnx.Linear(h_dim, 1, rngs = rngs)
    )

    return policy
    

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path)

    ## network parameters
    rngs = nnx.Rngs(0)
    h_dim = 128

    critic = build_pendulum_critic(h_dim, rngs)
    policy = build_pendulum_policy(h_dim, rngs)

    xt = jnp.zeros(2)
    xt = xt.at[0].set(3.1)

    print(xt)
    print(normalize_pendulum_state(xt))