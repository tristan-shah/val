'''
State-entropy (APT objective) MPC baseline on the fully actuated triple pendulum, the "APT (MPC)" column of Table 1.
Same planner and hyperparameters as the CIP run; only the objective differs.

    python scripts/state_entropy/experiments/triple_pendulum.py --seed 0

Writes traj.npy, U.npy, hist.npy, entropy.png and vid.mp4 to results/STATE_ENTROPY/TRIPLE_PENDULUM/<run>.
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
from val.state_entropy import make_compute_state_entropy


if __name__ == '__main__':

    parser = ArgumentParser(description = __doc__)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--k', type = int, default = 12, help = 'number of nearest neighbors')
    parser.add_argument('--subsample', type = int, default = 1, help = 'rollout thinning before the neighbor search')
    parser.add_argument('--horizon', type = int, default = 128)
    parser.add_argument('--shots', type = int, default = 2048)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--elite_frac', type = float, default = 0.1)
    parser.add_argument('--steps', type = int, default = 2000)
    parser.add_argument('--beta', type = float, default = 0.1, help = 'control penalty weight')
    parser.add_argument('--gear', type = float, default = 25.0, help = 'actuator gear (torque scale) of every joint')
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

    name = f'seed={seed}-k={args.k}-sub={args.subsample}-gear={args.gear}-beta={beta}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}-steps={steps}'
    path = Path('results/STATE_ENTROPY/TRIPLE_PENDULUM') / name
    path.mkdir(parents = True, exist_ok = True)

    ## dynamics with the gear override, and objective
    dyn = Dynamics('xml/triple_pendulum.xml', dt = dt)
    dyn.mjx_model = dyn.mjx_model.replace(
        actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(args.gear))
    step = make_step(dyn)
    compute_entropy = make_compute_state_entropy(step, args.k, args.subsample)

    def objective(xt, U):
        J, info = compute_entropy(xt, U)
        control_penalty = jnp.mean(jnp.sum(U ** 2, axis = 1))
        info['control_penalty'] = control_penalty
        return J - beta * control_penalty, info

    batch_objective = jax.jit(jax.vmap(objective, in_axes = (None, 0)))

    ## planner
    mpc = CEM(dyn, batch_objective, shots, horizon, iterations, elite_frac, smoothing, rho)

    ## hanging initial state (the model's zero pose is upright)
    xt = jnp.zeros(dyn.state_dim).at[0].set(jnp.pi)

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)
    hist = jnp.zeros(steps)
    controls = jnp.zeros((steps, dyn.control_dim))

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J, info = mpc(xt, subkey)
        xt = step(xt, ut)
        print(f'step {t:5d}  objective {float(J):.4f}', flush = True)

        controls = controls.at[t].set(ut)
        X = X.at[t + 1].set(xt)
        hist = hist.at[t].set(info['entropy'])

    jnp.save(path / 'hist.npy', hist)
    jnp.save(path / 'traj.npy', X)
    jnp.save(path / 'U.npy', controls)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State entropy (nats)')
    ax.plot(jnp.arange(0, steps) * dt, hist)
    fig.tight_layout()
    fig.savefig(path / 'entropy.png', dpi = 300)
    plt.close(fig)

    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 5)
