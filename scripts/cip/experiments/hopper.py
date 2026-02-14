# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
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
    steps = 2000
    iterations = 10
    elite_frac = 0.1
    smoothing = 0.1

    name = f'HOPPER-gear=25-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-dt={dt}'

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

    '''
    test passive dynamics
    '''
    # xt = jnp.array([0.23120155, -0.3827173,  -1.58013499, -1.35305359, -0.96104406,  0.72331184, 0.39360681, -0.73335341, -4.51459664, -3.6938281,  -2.78413389,  1.80420869])
    # xt = jnp.array([0.06307312, -0.63875374, -2.74784724, -1.73899677, -2.01111434,  0.79736904, -0.62935913, -0.44082156,  4.32641158,  5.58683695, -0.28963019,  0.05046539])
    # xt = jnp.array([8.05772871e-03, -5.93553076e-01, -3.05798126e+00, -2.10996381e+00, -1.81892265e+00,  7.99796965e-01,  1.43418526e+00, -1.21936469e+00, 8.81181065e+00,  8.72333283e+00, -3.74993284e+00, -4.03488516e-01])
    # xt = jnp.array([-0.13863229, -0.75210445, -2.8943634,  -1.45277164, -2.13336921,  0.69394748, 0.49754184,  0.48200094,  0.59733673, -1.1989103 ,  0.74639519,  1.11361288])
    # xt = jnp.array([0.07677816, -0.52848453, -3.32295781, -2.57912108, -1.63541612,  0.79667991, -0.17317498, -0.63602734, -2.85373577, -1.00127846, -2.92314251, -0.0769953 ])
    xt = jnp.array([-0.13162971, -0.94977474, -2.13520006, -0.10822481, -2.61761588,  0.48175869, -0.63438629,  0.30503823, -3.48250963, -3.02649539,  1.30653693, -1.62708323])

    unroll = make_unroll(step)
    U = jnp.zeros((horizon, dyn.control_dim))
    # U = jax.random.uniform(key, (horizon, dyn.control_dim)) * 2 - 1.0
    X = unroll(xt, U)
    dyn.render(X, path = 'hopper.mp4', skip = 1)


    # '''
    # run mpc
    # '''
    # X = jnp.zeros((steps + 1, dyn.state_dim))
    # X = X.at[0].set(xt)

    # hist = jnp.zeros(steps)

    # for t in range(steps):
    #     key, subkey = jax.random.split(key)
    #     ut, J = mpc(xt, subkey)
    #     xt = step(xt, ut)
    #     print(t, xt, ut, J)

    #     X = X.at[t+1].set(xt)
    #     hist = hist.at[t].set(J)

    # jnp.save(name + '-hist.npy', hist)
    # jnp.save(name + '-traj.npy', X)

    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('nats / s')
    # T = jnp.arange(0, steps)
    # ax.plot(T * dt, hist)
    # fig.tight_layout()
    # fig.savefig(name + '.png', dpi = 300)
    # plt.show()

    # dyn.render(X, path = name + '.mp4', skip = 1, distance = 4)