from pathlib import Path
import os
os.environ['MUJOCO_GL'] = 'egl'

from jax import numpy as jnp

from val import Dynamics

if __name__ == '__main__':
    
    dyn = Dynamics(path = 'xml/triple_pendulum.xml', dt = 0.01)

    root = Path('/mnt/CICI/home/trisshah/val/results/TRIPLE_PENDULUM/cip')
    # name = 'seed=0-gear=15.0-h=512-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    # name = 'seed=0-gear=15.0-h=1024-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    # name = 'seed=0-gear=15.0-h=1024-shots=1024-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    # name = 'seed=0-gear=15.0-h=2048-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    # name = 'seed=0-gear=25.0-h=1024-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    # name = 'seed=0-gear=25.0-h=1024-shots=1024-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    name = 'seed=0-gear=25.0-h=2048-shots=1024-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01'
    X = jnp.load(root / name / 'traj.npy')


    dyn.render(X, path = root / name / 'vid.mp4', skip = 1, distance = 5)