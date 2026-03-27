from pathlib import Path
from jax import numpy as jnp
import matplotlib.pyplot as plt

def load_average_history(base_path, method, num_seeds, horizon, dt):
    '''Load history arrays for multiple seeds and return stacked array.'''
    histories = []
    for seed in range(num_seeds):
        path = Path(f'{base_path}/{method}/seed={seed}-h={horizon}-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt={dt}')
        hist = jnp.load(path / 'hist.npy')
        histories.append(hist)
    return jnp.stack(histories).mean(axis = 0)  # shape: (num_seeds, time_steps, features)

if __name__ == '__main__':

    num_seeds = 6
    # root = 'results/SINGLE_PENDULUM/exponential_domain'

    ## task cartpole
    root = 'results/CART_POLE/exponential_domain'
    horizon = 400
    dt = 0.01

    cip_hist = load_average_history(root, 'cip', num_seeds, horizon, dt)
    ol_hist = load_average_history(root, 'ol', num_seeds, horizon, dt)

    print(cip_hist.mean(axis = 0))
    print(ol_hist.mean(axis = 0))


    # print(cl_hist.mean(axis = 0))

    # hist = jnp.load('/mnt/CICI/home/trisshah/val/results/CART_POLE/exponential_domain/cip/seed=0-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01/hist.npy')
    # print(hist)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(hist[:, 2])
    # fig.tight_layout()
    # fig.savefig('cl.png', dpi = 300)
    








    # T = 500
    # num_seeds = 5

    # cip_std = []
    # ol_std = []
    # for seed in range(num_seeds):

    #     cip = jnp.load(f'results/SINGLE_PENDULUM/cip/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/traj.npy')[T:]
    #     ol = jnp.load(f'results/SINGLE_PENDULUM/ol/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/traj.npy')[T:]

    #     cip_std.append(cip[:, 0].std().item())
    #     ol_std.append(ol[:, 0].std().item())

    #     print(cip_std[-1], ol_std[-1])


    # print()
    # print(f'Average of standard deviations across {num_seeds} random seeds:')
    # print(f'CIP: {jnp.mean(jnp.array(cip_std))}')
    # print(f'OL: {jnp.mean(jnp.array(ol_std))}')