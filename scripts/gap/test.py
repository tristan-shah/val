import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

if __name__ == '__main__':
    dt = 0.01
    dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
    steps = 500

    step = make_step(dyn)
    unroll = make_unroll(step)

    U = jnp.zeros((steps, dyn.control_dim))

    theta = 3.14
    x0 = jnp.zeros(dyn.state_dim)
    x0 = x0.at[0].set(theta)
    xt = x0.copy()

    X = unroll(xt, U)

    dyn.render(X, path = 'test.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))
