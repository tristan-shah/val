# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['MUJOCO_GL'] = 'egl'

from jax import numpy as jnp

from val import Dynamics, make_step, make_unroll


if __name__ == '__main__':
    

    dyn = Dynamics(path = 'xml/ant.xml', dt = 0.01)
    print(dyn.state_dim, dyn.control_dim)

    # unroll = make_unroll(make_step(dyn))

    ## initial state of ant
    # xt = jnp.array([
    #     4.63033074e-09, -1.33166722e-08,  5.88738445e-01,  1.00000000e+00,
    #     4.03076686e-10,  1.40153441e-10,  2.18994597e-19,  1.34050243e-09,
    #     1.08063526e+00, -6.48803570e-10, -1.08063526e+00, -1.34050247e-09,
    #     -1.08063526e+00,  6.48803545e-10,  1.08063526e+00,  1.69124988e-09,
    #     -4.86398140e-09, -2.83866190e-03,  7.92753681e-09,  2.75647608e-09,
    #     -6.05048348e-18,  4.30372771e-10, -1.06604418e-02, -2.08300861e-10,
    #     1.06604411e-02, -4.30372777e-10,  1.06604430e-02,  2.08300867e-10,
    #     -1.06604437e-02])

    # U = jnp.zeros((1000, dyn.control_dim))
    # X = unroll(xt, U)

    X = jnp.load('ANT-h=256-gamma=1.0-shots=128-iter=1-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt=0.01-traj.npy')

    dyn.render(X, path = 'ant.mp4', skip = 2)

    print(X.shape)