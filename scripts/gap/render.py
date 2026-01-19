from jax import numpy as jnp

from val import Dynamics

if __name__ == '__main__':
    

    dyn = Dynamics(path = 'xml/ant.xml', dt = 0.01)
    print(dyn.state_dim, dyn.control_dim)

    X = jnp.load('ANT-h=256-gamma=1.0-shots=128-iter=10-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt=0.01-traj.npy')
    # X = jnp.load('HOPPER-h=256-gamma=1.0-shots=512-iter=5-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt=0.01-traj.npy')

    dyn.render(X, path = 'ant.mp4', skip = 2)

    print(X.shape)