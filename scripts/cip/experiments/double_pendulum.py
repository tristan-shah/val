'''
CIP-driven MPC on the double pendulum (Section 4). The defaults are the paper's hyperparameters
(Table 3); one run is one seed.

    python scripts/cip/experiments/double_pendulum.py --seed 0

Writes traj.npy, U.npy, hist.npy, cip.png and vid.mp4 to results/CIP/DOUBLE_PENDULUM/<run>.
Run from the repository root; set MUJOCO_GL=egl for offscreen rendering on a headless machine.
'''

from argparse import ArgumentParser
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.cem import CEM
from val.cip import OBJECTIVES, RESULTS_FOLDER, make_compute_cip


if __name__ == '__main__':

    parser = ArgumentParser(description = __doc__)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--objective', type = str, default = 'cip', choices = OBJECTIVES,
                        help = "quantity to maximize: the open-loop rate 'cip' (paper), the negated 'closed_loop' rate, or their 'difference'")
    parser.add_argument('--horizon', type = int, default = 512)
    parser.add_argument('--shots', type = int, default = 1024)
    parser.add_argument('--iterations', type = int, default = 10)
    parser.add_argument('--elite_frac', type = float, default = 0.1)
    parser.add_argument('--steps', type = int, default = 2400)
    parser.add_argument('--beta', type = float, default = 0.0, help = 'control penalty weight (eta in Algorithm 1)')
    args = parser.parse_args()

    seed = args.seed
    key = jax.random.PRNGKey(seed)
    objective = args.objective

    dt = 0.01
    horizon = args.horizon
    shots = args.shots
    steps = args.steps
    iterations = args.iterations
    elite_frac = args.elite_frac
    smoothing = 0.1
    rho = 0.9
    beta = args.beta

    name = f'seed={seed}-beta={beta}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}-steps={steps}'
    path = Path('results') / RESULTS_FOLDER[objective] / 'DOUBLE_PENDULUM' / name
    path.mkdir(parents = True, exist_ok = True)

    ## dynamics and objective
    dyn = Dynamics('xml/double_pendulum.xml', dt = dt, integrator = 'implicitfast')
    step = make_step(dyn)
    compute_cip = make_compute_cip(dyn, objective)

    def penalized_objective(xt, U):
        J, info = compute_cip(xt, U)
        control_penalty = jnp.mean(jnp.sum(U ** 2, axis = 1))
        info['control_penalty'] = control_penalty
        return J - beta * control_penalty, info

    batch_objective = jax.jit(jax.vmap(penalized_objective, in_axes = (None, 0)))

    ## planner
    mpc = CEM(dyn, batch_objective, shots, horizon, iterations, elite_frac, smoothing, rho)

    ## hanging initial state
    xt = jnp.zeros(dyn.state_dim)

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)
    hist = jnp.zeros((steps, 3))
    controls = jnp.zeros((steps, dyn.control_dim))

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J, info = mpc(xt, subkey)
        xt = step(xt, ut)
        print(f'step {t:5d}  objective {float(J):.4f}', flush = True)

        controls = controls.at[t].set(ut)
        X = X.at[t + 1].set(xt)
        ## hist columns: (difference, CIP, closed-loop rate), the order used by all existing runs
        hist = hist.at[t].set(jnp.array([info['difference'], info['cip'], info['closed_loop']]))

    jnp.save(path / 'hist.npy', hist)
    jnp.save(path / 'traj.npy', X)
    jnp.save(path / 'U.npy', controls)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist[:, 1], label = 'CIP (open-loop rate)')
    ax.plot(T * dt, hist[:, 2], label = 'Closed-loop rate')
    ax.plot(T * dt, hist[:, 0], label = 'Difference')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'cip.png', dpi = 300)
    plt.close(fig)

    dyn.render(X, path = path / 'vid.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))
