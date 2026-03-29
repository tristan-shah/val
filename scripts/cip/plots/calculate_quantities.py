from pathlib import Path
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step

def load_average_history(base_path, method, num_seeds, horizon, dt):
    '''Load history arrays for multiple seeds and return stacked array.'''
    histories = []
    for seed in range(num_seeds):
        path = Path(f'{base_path}/{method}/seed={seed}-h={horizon}-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt={dt}')
        hist = jnp.load(path / 'hist.npy')
        histories.append(hist)
    return jnp.stack(histories).mean(axis = 0)  # shape: (num_seeds, time_steps, features) <- average over seeds

if __name__ == '__main__':

    # num_seeds = 6
    # ## task cartpole
    # root = 'results/CIP/CART_POLE/exponential_domain'
    # horizon = 400
    dt = 0.01

    # cip_hist = load_average_history(root, 'cip', num_seeds, horizon, dt)
    # ol_hist = load_average_history(root, 'ol', num_seeds, horizon, dt)

    # print(cip_hist.mean(axis = 0))
    # print(ol_hist.mean(axis = 0))

    path = Path('/Users/tristanshah/Desktop/code/val/results/CIP/CART_POLE/exponential_domain/cip/seed=0-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01/traj.npy')
    cip_X = jnp.load(path)

    print(cip_X)

    dyn = Dynamics('xml/cart_pole.xml', dt = dt)
    step = make_step(dyn)
    
    dyn.render(cip_X, path =  'vid.mp4', skip = 2, distance = 4)