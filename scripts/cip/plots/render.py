from pathlib import Path
from jax import numpy as jnp

from val import Dynamics

if __name__ == '__main__':
    
    dyn = Dynamics(path = 'xml/triple_pendulum_white.xml', dt = 0.01)

    root = Path('results/TRIPLE_PENDULUM/ol')
    name = 'retest-seed=7-gear=25.0-beta=2.5-h=128-shots=2048-iter=2-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200'
    X = jnp.load(root / name / 'traj.npy')

    dyn.render(X, path = root / name / 'white_vid.mp4', skip = 1, distance = 5)