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
    print(dyn.state_dim, dyn.control_dim)

    step = make_step(dyn)
    
    xt = jnp.array([
        4.63033074e-09, -1.33166722e-08,  5.88738445e-01,  1.00000000e+00,
        4.03076686e-10,  1.40153441e-10,  2.18994597e-19,  1.34050243e-09,
        1.08063526e+00, -6.48803570e-10, -1.08063526e+00, -1.34050247e-09,
        -1.08063526e+00,  6.48803545e-10,  1.08063526e+00,  1.69124988e-09,
        -4.86398140e-09, -2.83866190e-03,  7.92753681e-09,  2.75647608e-09,
        -6.05048348e-18,  4.30372771e-10, -1.06604418e-02, -2.08300861e-10,
        1.06604411e-02, -4.30372777e-10,  1.06604430e-02,  2.08300867e-10,
        -1.06604437e-02])

    unroll = make_unroll(step)

    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]

    U = jnp.zeros((horizon, dyn.control_dim))# + jax.random.normal(key, (horizon, dyn.control_dim)) * 0.1
    U = U.clip(low, high)
    X = unroll(xt, U)

    print(X[-1])
    dyn.render(X, path = 'ant.mp4', skip = 1)

