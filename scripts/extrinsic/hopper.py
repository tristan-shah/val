import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.cem import CEM

def hopper_initial_state():
    return jnp.array([-2.0, -0.245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def make_compute_hopper_cost(dyn: Dynamics, goal_x: float = 5.0):

    unroll = make_unroll(make_step(dyn))

    def double_pendulum_cost(xt: Array, U: Array):
        X = unroll(xt, U)
        J = jnp.sum((X[:, 0] - goal_x) ** 2)
        return -J

    return jax.jit(double_pendulum_cost)

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)

    dt = 0.01
    horizon = 200
    shots = 128
    steps = 1000
    iterations = 1
    elite_frac = 0.1
    smoothing = 0.1
    rho = 0.9
    gear = 25

    name = f'HOPPER-gear={gear}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}'

    dyn = Dynamics('xml/hopper.xml', dt = dt)

    ## override default gear strength
    dyn.mjx_model = dyn.mjx_model.replace(
        actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(gear)
    )

    step = make_step(dyn)
    print(dyn.state_dim, dyn.control_dim)

    objective = make_compute_hopper_cost(dyn, goal_x = 2.0)
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
    xt = hopper_initial_state()

    # '''
    # passive dynamics
    # '''
    # U = jnp.zeros((horizon, dyn.control_dim))
    # unroll = make_unroll(step)
    # X = unroll(xt, U)
    # dyn.render(X, path = name + '.mp4', skip = 2, distance = 6.0)





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
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(name + '.png', dpi = 300)
    plt.show()

    dyn.render(X, path = name + '.mp4', skip = 2, distance = 6.0)