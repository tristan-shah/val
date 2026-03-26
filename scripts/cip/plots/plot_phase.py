from jax import numpy as jnp

if __name__ == '__main__':

    T = 500
    num_seeds = 5

    cip_std = []
    ol_std = []
    for seed in range(num_seeds):

        cip = jnp.load(f'results/SINGLE_PENDULUM/cip/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/traj.npy')[T:]
        ol = jnp.load(f'results/SINGLE_PENDULUM/ol/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/traj.npy')[T:]

        cip_std.append(cip[:, 0].std().item())
        ol_std.append(ol[:, 0].std().item())

        print(cip_std[-1], ol_std[-1])


    print()
    print(f'Average of standard deviations across {num_seeds} random seeds:')
    print(f'CIP: {jnp.mean(jnp.array(cip_std))}')
    print(f'OL: {jnp.mean(jnp.array(ol_std))}')