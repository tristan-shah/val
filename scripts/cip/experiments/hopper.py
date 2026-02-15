import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.cem import CEM
from val.cip import make_compute_cip

def hopper_initial_state():
    return jnp.array([0.0, -0.245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    dt = 0.01
    horizon = 512
    shots = 512
    steps = 500
    iterations = 10
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9

    gear = 100
    name = f'HOPPER-gear={gear}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'

    ## initialize dynamics
    dyn = Dynamics('xml/hopper.xml')
    
    step = make_step(dyn)

    compute_cip = make_compute_cip(dyn)
    ## vectorize over batches of trajectories
    batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes = (None, 0)))

    ## initialize agent
    mpc = CEM(
        dyn,
        batch_compute_cip,
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        smoothing)
    
    ## get initial state
    xt = hopper_initial_state()

    print(dyn.control_dim)

    # '''
    # test passive dynamics
    # '''
    # # xt = jnp.array([-1.43900763, -0.73017787, -6.83050736, -0.01042726, -2.47514614,  0.51961674, 0.48878844, -0.43453321,  0.06770939, -1.50383176, -0.18066599,  1.08577407])
    # # xt = jnp.array([-0.80104101, -0.95588666, -7.77412811, -2.48870395, -1.08208366,  0.03185777, -0.62704375,  0.77284243,  1.42728638,  3.46597395, -3.21686201,  1.84237974])
    # # xt = jnp.array([-1.38053211, -0.75585558, -6.98581127, -0.34440876, -2.43059323,  0.6027617, 1.2667581,  -0.42306029, -4.14837958, -7.70671066,  1.27446845,  1.15476878])
    # # xt = jnp.array([-1.17430685, -0.66986028, -8.46305291, -2.16715771, -1.89754507,  0.78775427, -0.1043199,  -0.97346467,  5.77077936,  6.90764519, -4.02079525, -0.02615875])
    # # xt = jnp.array([-0.52359518, -1.01024797, -7.78697348, -2.64011594, -0.90791346, -0.11269569, -0.78527831,  1.39504878, -5.60225903, -2.21421301, -4.45918144,  5.70114674])
    # # xt = jnp.array([-0.70922936, -1.17690323, -6.71164888, -1.94839871, -0.53250935, -0.08287912, -0.04711833,  0.11924502,  3.27362498,  3.56487311, -1.08397781,  1.15760398])
    # # xt = jnp.array([-0.18679878, -0.77642749, -2.99706669, -1.43937731, -2.15453168,  0.59842824, 0.55295064,  0.44344653,  2.84606065,  1.33817209,  0.61349928,  0.87457951])

    # from val import make_unroll
    # unroll = make_unroll(step)
    # # U = jnp.zeros((horizon, dyn.control_dim))
    # U = jax.random.uniform(key, (horizon, dyn.control_dim)) * 2.0 - 1.0
    # X = unroll(xt, U)
    # dyn.render(X, path = 'hopper.mp4', skip = 1)


    '''
    run mpc
    '''
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros(steps)

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J = mpc(xt, subkey)
        xt = step(xt, ut)
        print(t, xt, ut, J)

        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(J)

    jnp.save(name + '-hist.npy', hist)
    jnp.save(name + '-traj.npy', X)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(name + '.png', dpi = 300)
    plt.show()

    dyn.render(X, path = name + '.mp4', skip = 1, distance = 4)