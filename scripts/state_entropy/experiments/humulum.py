'''
State-entropy (APT objective) MPC baseline on the gibbon (planar humanoid hanging by its feet), the "APT (MPC)" column of Table 1.
Same planner as the CIP run; only the objective (and the control penalty) differs. The run
starts from the settled hanging pose in xml/hanging.npy after a few warm-start planning rounds.

    python scripts/state_entropy/experiments/humulum.py --seed 0

Writes traj.npy, U.npy, hist.npy, entropy.png and vid.mp4 to results/STATE_ENTROPY/HUMULUM/<run>.
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
    parser.add_argument('--horizon', type = int, default = 512)
    parser.add_argument('--shots', type = int, default = 1024)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--elite_frac', type = float, default = 0.2)
    parser.add_argument('--steps', type = int, default = 1200)
    parser.add_argument('--beta', type = float, default = 1.0, help = 'control penalty weight')
    parser.add_argument('--warmstart', type = int, default = 10, help = 'planning rounds before the episode starts')
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

    name = f'seed={seed}-k={args.k}-sub={args.subsample}-beta={beta}-h={horizon}-shots={shots}-iter={iterations}-elite={elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}-steps={steps}'
    path = Path('results/STATE_ENTROPY/HUMULUM') / name
    path.mkdir(parents = True, exist_ok = True)

    ## dynamics and objective
    dyn = Dynamics('xml/humulum.xml', dt = dt)
    step = make_step(dyn)
    compute_entropy = make_compute_state_entropy(step, args.k, args.subsample)

    def penalized_objective(xt, U):
        J, info = compute_entropy(xt, U)
        control_penalty = jnp.mean(jnp.sum(U ** 2, axis = 1))
        info['control_penalty'] = control_penalty
        return J - beta * control_penalty, info

    batch_objective = jax.jit(jax.vmap(penalized_objective, in_axes = (None, 0)))

    ## planner
    mpc = CEM(dyn, batch_objective, shots, horizon, iterations, elite_frac, smoothing, rho)

    ## settled hanging pose
    xt = jnp.load('xml/hanging.npy')

    ## warm-start the sampling distribution without advancing the state
    for i in range(args.warmstart):
        key, subkey = jax.random.split(key)
        _, J, _ = mpc(xt, subkey, roll = False)
        print(f'warmstart {i:3d}  objective {float(J):.4f}', flush = True)

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

    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 5, lookat = jnp.array([0.0, 0.0, 0.0]))
