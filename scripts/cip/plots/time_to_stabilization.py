from pathlib import Path
from jax import numpy as jnp

if __name__ == '__main__':

    root = Path('results/CIP/DOUBLE_PENDULUM/ol')

    seed = 0
    path = root / f'seed={seed}-beta=0.0-h=512-shots=1024-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=2400/traj.npy'

    X = jnp.load(path)

    print(X)