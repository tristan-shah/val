from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.cem import CEM
from val.cip import make_compute_cip

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--component', type = str, default = 'ol')
    parser.add_argument('--horizon', type = int, default = 512)
    parser.add_argument('--shots', type = int, default = 1024)
    parser.add_argument('--gear', type = float, default = None)
    parser.add_argument('--damping', type = float, default = None)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--elite_frac', type = float, default = 0.2)
    parser.add_argument('--steps', type = int, default = 1200)
    parser.add_argument('--beta', type = float, default = 9.0)
    parser.add_argument('--warmstart', type = int, default = 10)
    args = parser.parse_args()

    seed = args.seed
    key = jax.random.PRNGKey(seed)

    component = args.component

    dt = 0.01
    horizon = args.horizon
    shots = args.shots
    steps = args.steps
    iterations = args.iterations
    elite_frac = args.elite_frac
    smoothing = 0.1
    rho = 0.9
    beta = args.beta
    warmstart = args.warmstart

    name = f'seed={seed}-gear={args.gear}-damp={args.damping}-warmstart={warmstart}-beta={beta}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}-steps={args.steps}'
    root = Path(f'results/HUMULUM/efficient/{component}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    dyn = Dynamics('xml/humulum.xml', dt = dt)
    print(f'State Dim {dyn.state_dim}, Control Dim {dyn.control_dim}')

    if args.gear is not None:
        ## override default gear strength
        dyn.mjx_model = dyn.mjx_model.replace(
            actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(args.gear)
        )

    if args.damping is not None:
        dyn.mjx_model = dyn.mjx_model.replace(
            dof_damping = dyn.mjx_model.dof_damping.at[:].set(args.damping)
        )


    step = make_step(dyn)

    '''
    running experiment
    '''
    compute_cip = make_compute_cip(dyn, component)

    def objective(xt, U):
        J, info = compute_cip(xt, U)
        control_penalty = jnp.mean(jnp.sum(U ** 2, axis = 1))
        info['control_penalty'] = control_penalty
        return J - beta * control_penalty, info

    batch_objective = jax.jit(jax.vmap(objective, in_axes = (None, 0)))

    ## initialize agent
    mpc = CEM(
        dyn,
        batch_objective,
        shots,
        horizon, 
        iterations, 
        elite_frac,
        smoothing,
        rho)

    ## load in hanging pose
    xt = jnp.load('xml/hanging.npy')

    ut = jnp.zeros(dyn.control_dim)

    linearize = jax.jacfwd(step)
    fx = linearize(xt, ut)

    print(fx.min(), fx.max())



    ## warmstart
    for i in range(args.warmstart):
        key, subkey = jax.random.split(key)
        ut, J, info, U = mpc(xt, subkey, roll = False)
        print(i, J)


    '''
    Run MPC
    '''
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros((steps, 3))
    controls = jnp.zeros((steps, dyn.control_dim))

    for t in range(steps):

        key, subkey = jax.random.split(key)
        ut, J, info, U = mpc(xt, subkey)

        xt = step(xt, ut)
        print(t, xt, ut, J)

        controls = controls.at[t].set(ut)
        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(jnp.array([info['cip'], info['ol'], info['cl']]))

    jnp.save(path / 'hist.npy', hist)
    jnp.save(path / 'traj.npy', X)
    jnp.save(path / 'U.npy', controls)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist[:, 0], label = 'CIP')
    ax.plot(T * dt, hist[:, 1], label = 'OL')
    ax.plot(T * dt, hist[:, 2], label = 'CL')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'metrics.png', dpi = 300)
    plt.show()

    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 5, lookat = jnp.array([0.0, 0.0, 0.0]))