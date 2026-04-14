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
[ 0.12226839 -0.06047546 10.91040093 -0.46536473 -0.16035833  6.96815383
 -0.34869982 -0.19139979  0.73149846 -0.40931179 -6.51437417 13.39808567
 -6.88085735 -0.13750666 -1.69418634 20.07478373  9.96628679 -1.30582379
  5.77337816  0.70600948 -8.29083329 17.20760134 -8.03732672 -6.13909785
 12.71443837 -7.53413667]
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