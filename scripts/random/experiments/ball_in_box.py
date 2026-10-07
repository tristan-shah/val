'''
Uniform-random control baseline for the ball-in-box environment (Appendix A.1, Figure 6).

No planner and no objective: at every step the control is drawn uniformly from the actuator
range. Saves traj.npy and U.npy in the same format as the CIP run to results/RANDOM/BALL_IN_BOX/<run>.

    python scripts/random/experiments/ball_in_box.py --restitution 0.5 --seed 0
'''

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import jax
from jax import numpy as jnp

from val.ball import BallInBox
from val.dynamics import make_unroll


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--steps', type = int, default = 1000)
    parser.add_argument('--restitution', type = float, default = 0.5, help = 'wall restitution coefficient')
    parser.add_argument('--force_limit', type = float, default = 1.0)
    parser.add_argument('--random_spawn', action = BooleanOptionalAction, default = True,
                        help = 'start at a uniformly random position (otherwise the center)')
    parser.add_argument('--render', action = 'store_true', help = 'also write vid.mp4')
    args = parser.parse_args()

    seed = args.seed
    key = jax.random.PRNGKey(seed)

    dt = 0.05
    steps = args.steps
    restitution = args.restitution

    name = f'seed={seed}-e={restitution}-spawn={args.random_spawn}-dt={dt}-steps={steps}'
    root = Path('results/RANDOM/BALL_IN_BOX')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    ## initialize dynamics
    dyn = BallInBox(dt = dt, restitution = restitution, force_limit = args.force_limit)
    unroll = make_unroll(dyn.step)

    ## initial state: ball at rest, at the center or (optionally) a random position
    xt = jnp.zeros(dyn.state_dim)
    if args.random_spawn:
        xmin, xmax, ymin, ymax = dyn.bounds
        key, subkey = jax.random.split(key)
        p0 = jax.random.uniform(
            subkey, (2,),
            minval = jnp.array([xmin, ymin]),
            maxval = jnp.array([xmax, ymax]))
        xt = xt.at[:2].set(p0)

    ## pure uniform-random control over the whole episode, then roll out
    key, subkey = jax.random.split(key)
    U = jax.random.uniform(
        subkey, (steps, dyn.control_dim),
        minval = dyn.low[None, :], maxval = dyn.high[None, :])
    X = unroll(xt, U)                              # (steps + 1, state_dim)

    jnp.save(path / 'traj.npy', X)
    jnp.save(path / 'U.npy', U)

    if args.render:
        dyn.render(X, path = path / 'vid.mp4', skip = 1)
