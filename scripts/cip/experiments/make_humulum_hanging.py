import os

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

import re

def parse_state_string(s):
    numbers = re.findall(r'[-+]?\d+\.?\d*(?:e[+-]?\d+)?', s)
    return jnp.array([float(n) for n in numbers])

state = '''
[ 5.63351668e-02 -1.39881434e+00  1.42891115e+02 -1.25796109e+00                                                                                            
  2.47824753e+00  1.53936631e-03  2.50630286e+00 -3.07993262e+00                                                                                                 
  5.99858672e+00 -2.92234212e+00 -3.25391053e+00  6.33964957e+00                                                                                                 
 -3.05710620e+00  3.30682258e-01  2.17950948e-01  1.20418214e+01                                                                                                 
 -1.43610785e+00  9.84460215e-01  4.49690191e+00 -8.78519212e-01                                                                                                 
  1.56709767e+00 -4.11787557e+00  5.34675091e+00 -1.41004742e+00                                                                                                 
  2.19462547e+00  1.69740172e+00]
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