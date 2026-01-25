from jax import numpy as jnp

from val import Dynamics

if __name__ == '__main__':
    

    # dyn = Dynamics(path = 'xml/ant.xml', dt = 0.01)
    dyn = Dynamics(path = 'xml/double_pendulum.xml', dt = 0.01)
    # dyn = Dynamics(path = 'xml/pendulum.xml', dt = 0.01)
    # dyn = Dynamics(path = 'xml/cart_pole.xml', dt = 0.01)
    # dyn = Dynamics(path = 'xml/hopper.xml', dt = 0.01)
    print(dyn.state_dim, dyn.control_dim)

    # name = 'HOPPER-gear=25-h=256-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
    name = 'DOUBLE_PENDULUM-gear=6.0-h=512-gamma=1.0-shots=512-iter=10-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
    # name = 'CART_POLE-h=400-gamma=1.0-shots=512-iter=1-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
    # name = 'SINGLE_PENDULUM-h=600-gamma=1.0-shots=512-iter=1-elite=0.1-smooth=0.1-alpha=1.0-dt=0.01'
    X = jnp.load(name + '-traj.npy')

    # dyn.render(X, path = name + '.mp4', skip = 3)
    dyn.render(X, path = name + '.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))