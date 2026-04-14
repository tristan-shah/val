import os

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

import re

# def parse_state_string(s):
#     numbers = re.findall(r'[-+]?\d+\.\d+e[+-]\d+', s)
#     return jnp.array([float(n) for n in numbers])

def parse_state_string(s):
    numbers = re.findall(r'[-+]?\d+\.?\d*(?:e[+-]?\d+)?', s)
    return jnp.array([float(n) for n in numbers])


state = '''
[ 2.04514046e-01 -4.99051551e-02  1.75138192e+01  5.82318228e+00
  1.86790795e-02  1.35129091e+01  2.34620763e-01  5.03735935e-01
 -4.18172488e-01 -4.19941034e-03 -5.72992211e+00  1.20599177e+01
 -6.29435774e+00 -2.20917723e-01  3.74325632e-01  9.63302499e+00
  1.57981503e+00  1.24819013e+00  3.84473741e+00 -1.61381353e-01
 -1.92184123e+00  3.08015708e+00 -1.28875613e+00 -2.06217524e+00
  3.59359908e+00 -1.39506556e+00]
'''

if __name__ == '__main__':

    horizon = 512
    dt = 0.01
    dyn = Dynamics('xml/humulum.xml', dt = dt)
    print(f'State Dim {dyn.state_dim}, Control Dim {dyn.control_dim}')
    step = make_step(dyn)

    '''
    Inspecting Jacobians
    '''
    unroll = make_unroll(step)
    # xt = jnp.zeros(dyn.state_dim)
    xt = parse_state_string(state)


    U = jnp.zeros((horizon, dyn.control_dim))
    X = unroll(xt, U)

    dyn.render(X, path = 'vid.mp4', skip = 1, distance = 5, lookat = jnp.array([0.0, 0.0, 0.0]))
    # jnp.save('hanging.npy', X[-1])