# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM
from val.utils import smooth_angle_wrap

@jax.jit
def compute_double_pendulum_error(X: Array):

    r = jnp.stack([
        smooth_angle_wrap(X[:, 0] - jnp.pi),
        smooth_angle_wrap(X[:, 0] + X[:, 1] - jnp.pi),
        X[:, 2] * 0,
        X[:, 3] * 0
    ], axis = 1)

    return r

def make_compute_double_pendulum_cost(dyn: Dynamics):

    unroll = make_unroll(make_step(dyn))

    def double_pendulum_cost(xt: Array, U: Array):
        X = unroll(xt, U)
        r = compute_double_pendulum_error(X)
        J = jnp.sum(jnp.square(r))
        return -J

    return jax.jit(double_pendulum_cost)

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    dt = 0.01
    horizon = 512
    shots = 128
    steps = 2000
    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9

    name = f'DOUBLE_PENDULUM-gear=6.0-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'

    dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
    step = make_step(dyn)
    print(dyn.state_dim, dyn.control_dim)

    objective = make_compute_double_pendulum_cost(dyn)
    batch_objective = jax.jit(jax.vmap(objective, in_axes = (None, 0)))

    mpc = CEM(
        dyn,
        batch_objective,
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        smoothing,
        rho)
    
    ## initial state
    xt = jnp.zeros(dyn.state_dim)

    # '''
    # passive dynamics
    # '''
    # xt = xt.at[0].set(0)
    # U = jnp.zeros((horizon, dyn.control_dim))
    # # from val.dynamics import make_unroll
    # # unroll = make_unroll(step)
    # # X = unroll(xt, U)
    # # dyn.render(X, path = 'test.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))
    # print(compute_double_pendulum_cost(xt, U))

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

    dyn.render(X, path = name + '.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))