import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    # dt = 0.01 ## double
    # dt = 0.05 ## single
    # horizon = 1000
    horizon = 1000 ## single
    shots = 256
    eps = 0.2
    iterations = 5
    steps = 600 ## single
    # steps = 3000 ## double
    window = 1

    dyn = Dynamics('xml/ant.xml')
    # dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
    step = make_step(dyn)

    xt = jnp.concatenate([dyn.mjx_model.numeric_data, jnp.zeros(dyn.nv)])

    xt = dyn.init_state()

    unroll = make_unroll(step)


    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]

    U = jnp.zeros((horizon, dyn.control_dim)) + jax.random.normal(key, (horizon, dyn.control_dim)) * 0.5
    U = U.clip(low, high)
    X = unroll(xt, U)

    dyn.render(X, path = 'ant.mp4', skip = 1)