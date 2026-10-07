"""
Renders a white-background video of one triple pendulum CIP run for the project webpage
(not in the paper). Run from the repository root:

    python scripts/webpage/video_triple_pendulum.py --seed 7
"""

from argparse import ArgumentParser
from pathlib import Path

from jax import numpy as jnp

from val import Dynamics

ROOT = Path('results/CIP/TRIPLE_PENDULUM')
PAT  = 'seed={s}-gear=25.0-beta=2.5-h=128-shots=2048-iter=2-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200'

if __name__ == '__main__':

    parser = ArgumentParser(description = __doc__)
    parser.add_argument('--seed', type = int, default = 7)
    args = parser.parse_args()

    dyn = Dynamics(path = 'xml/triple_pendulum_white.xml', dt = 0.01)

    run = ROOT / PAT.format(s = args.seed)
    X = jnp.load(run / 'traj.npy')

    dyn.render(X, path = run / 'white_vid.mp4', skip = 1, distance = 5)
    print(f'Saved {run / "white_vid.mp4"}')
