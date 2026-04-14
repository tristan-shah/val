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
[-3.96654778e-03 -7.03107841e-01 -1.80799282e+01 -1.08881612e+01
 -4.14065508e-01  2.08264939e+00 -5.53295326e-01 -1.42922446e+00
  9.43933612e+00 -8.02812077e+00 -9.23338496e+00  9.39442976e+00
 -1.24130978e-01  4.52052962e-02  6.67653757e-02 -3.60784702e+00
  4.91838411e+00 -4.97035196e-01  3.99895729e+00  2.41390778e+00
  1.61004152e+01 -3.93051813e-01 -1.62930853e+01 -2.51520371e+00
 -5.73284423e-01  3.47847708e+00]
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