from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.cem import CEM
from entropy_objective import make_compute_state_entropy

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--k', type = int, default = 12)
    parser.add_argument('--subsample', type = int, default = 1)
    parser.add_argument('--horizon', type = int, default = 128)
    parser.add_argument('--shots', type = int, default = 2048)
    parser.add_argument('--gear', type = float, default = 25)
    parser.add_argument('--damping', type = float, default = None)
    parser.add_argument('--iterations', type = int, default = 10)
    parser.add_argument('--elite_frac', type = float, default = 0.1)
    parser.add_argument('--steps', type = int, default = 1200)
    parser.add_argument('--beta', type = float, default = 2.5)
    args = parser.parse_args()

    seed = args.seed
    key = jax.random.PRNGKey(seed)

    dt = 0.01
    horizon = args.horizon
    shots = args.shots
    steps = args.steps
    iterations = args.iterations
    elite_frac = args.elite_frac
    smoothing = 0.1
    rho = 0.9
    beta = args.beta
    k = args.k
    subsample = args.subsample

    name = f'retest-seed={seed}-k={k}-sub={subsample}-gear={args.gear}-damp={args.damping}-beta={beta}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}-steps={args.steps}'
    root = Path('results/STATE_ENTROPY/TRIPLE_PENDULUM')

    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    dyn = Dynamics('xml/triple_pendulum.xml', dt = dt)

    ## override default gear strength
    dyn.mjx_model = dyn.mjx_model.replace(
        actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(args.gear)
    )

    if args.damping is not None:
        dyn.mjx_model = dyn.mjx_model.replace(
            dof_damping = dyn.mjx_model.dof_damping.at[:].set(args.damping)
        )

    print(dyn.mjx_model.actuator_gear)
    print(dyn.mjx_model.dof_damping)

    step = make_step(dyn)

    compute_se = make_compute_state_entropy(step, k, subsample)

    def objective(xt, U):
        J, info = compute_se(xt, U)
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

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(jnp.pi)

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
    ax.set_ylabel('nats')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist[:, 0], label = 'State Entropy (nats)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'metrics.png', dpi = 300)
    plt.show()

    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 5)
