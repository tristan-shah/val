from pathlib import Path
from jax import numpy as jnp

import matplotlib.pyplot as plt

from val import Dynamics

def chunk_episode(root: Path, horizon: int = 10) -> list[dict]:
    '''Load a single episode and return a list of (obs, actions) chunks.'''

    X = jnp.load(root / 'traj.npy')
    U = jnp.load(root / 'U.npy')

    T = U.shape[0]

    obs = []
    actions = []

    for i in range(T - horizon):
        obs.append(X[i])
        actions.append(U[i:i+horizon])

    return jnp.stack(obs), jnp.stack(actions)

if __name__ == '__main__':

    ## load in an episode
    root = Path('/Users/tristanshah/Desktop/code/val/results/HUMULUM/efficient/ol/seed=0-beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200/')
    # obs, actions = chunk_episode(root, horizon = 10)
    # print(obs.shape, actions.shape)

    X = jnp.load(root / 'traj.npy')


    dt = 0.01
    dyn = Dynamics('xml/humulum.xml', dt = dt)
    
    print(X)
    print(dyn)