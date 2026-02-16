from jax import numpy as jnp

from val import Dynamics

if __name__ == '__main__':
    

    # dyn = Dynamics(path = 'xml/ant.xml', dt = 0.01)
    # dyn = Dynamics(path = 'xml/double_pendulum.xml', dt = 0.01)
    # dyn = Dynamics(path = 'xml/pendulum.xml', dt = 0.01)
    # dyn = Dynamics(path = 'xml/cart_pole.xml', dt = 0.01)
    dyn = Dynamics(path = 'xml/unrestricted_hopper.xml', dt = 0.01)
    print(dyn.state_dim, dyn.control_dim)
    
    name = 'UNRESTRICTED_HOPPER-gear=50-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    X = jnp.load(name + '-traj.npy')

    dyn.render(X, path = name + '.mp4', skip = 2, distance = 6.0)
    # dyn.render(X, path = name + '.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))