
from jax import Array
from jax import numpy as jnp

from val import Dynamics, make_step
from val.cip import make_compute_cip

def print_passive_ol(dyn: Dynamics, xt: Array):
    compute_cip = make_compute_cip(dyn, 'ol')
    U = jnp.zeros((512, dyn.control_dim))
    print(compute_cip(xt, U))
    return None

if __name__ == '__main__':
    
    dt = 0.01


    '''
    cart pole
    '''
    ## initialize dynamics
    dyn = Dynamics('xml/cart_pole.xml', dt = dt)
    xt = jnp.zeros(dyn.state_dim)
    print_passive_ol(dyn, xt)

    '''
    double pendulum
    '''
    dyn = Dynamics('xml/double_pendulum.xml', dt = dt, integrator = 'implicitfast')
    xt = jnp.zeros(dyn.state_dim)
    print_passive_ol(dyn, xt)

    '''
    triple pendulum
    '''
    dyn = Dynamics('xml/triple_pendulum.xml', dt = dt)
        ## override default gear strength
    dyn.mjx_model = dyn.mjx_model.replace(
        actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(25)
    )
    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(jnp.pi)
    print_passive_ol(dyn, xt)

    '''
    gibbon pendulum
    '''
    dyn = Dynamics('xml/humulum.xml', dt = dt)
    xt = jnp.load('hanging.npy')
    print_passive_ol(dyn, xt)