# from jax import numpy as jnp
# import matplotlib.pyplot as plt


# if __name__ == '__main__':

#     # cip_U = jnp.load('/mnt/CICI/home/trisshah/val/results/SINGLE_PENDULUM/exponential_domain/cip/seed=0-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/U.npy')
#     # ol_U = jnp.load('/mnt/CICI/home/trisshah/val/results/SINGLE_PENDULUM/exponential_domain/ol/seed=0-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/U.npy')

#     seed = 1

#     cip_U = jnp.load(f'/mnt/CICI/home/trisshah/val/results/SINGLE_PENDULUM/exponential_domain/cip/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/U.npy')
#     ol_U = jnp.load(f'/mnt/CICI/home/trisshah/val/results/SINGLE_PENDULUM/exponential_domain/ol/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/U.npy')

#     print('Full episode')
#     print(f'CIP: {jnp.mean(jnp.square(cip_U))}')
#     print(f'OL: {jnp.mean(jnp.square(ol_U))}')

#     print()

#     print('Last half')
#     print(f'CIP: {jnp.mean(jnp.square(cip_U[300:]))}')
#     print(f'OL: {jnp.mean(jnp.square(ol_U[300:]))}')


#     fig, ax = plt.subplots(1, 1)
#     ax.set_title('True Actions Applied in the Dynamics')
#     ax.set_xlabel('Timestep')
#     ax.set_ylabel('Control')
#     ax.plot(cip_U, label = 'CIP')
#     ax.plot(ol_U, label = 'OL')
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig('u.png', dpi = 300)


import os
os.environ["JAX_PLATFORMS"] = "cpu"   # optional: avoids CUDA init warning

from jax import numpy as jnp
import matplotlib.pyplot as plt


if __name__ == '__main__':

    cip_full_vals = []
    ol_full_vals = []

    cip_half_vals = []
    ol_half_vals = []

    cip_all_U = []
    ol_all_U = []

    for seed in range(6):  # seeds 0..5
        cip_U = jnp.squeeze(jnp.load(
            f'/mnt/CICI/home/trisshah/val/results/SINGLE_PENDULUM/exponential_domain/cip/'
            f'seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/U.npy'
        ))
        ol_U = jnp.squeeze(jnp.load(
            f'/mnt/CICI/home/trisshah/val/results/SINGLE_PENDULUM/exponential_domain/ol/'
            f'seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/U.npy'
        ))

        cip_all_U.append(cip_U)
        ol_all_U.append(ol_U)

        cip_full_vals.append(jnp.mean(jnp.square(cip_U)))
        ol_full_vals.append(jnp.mean(jnp.square(ol_U)))

        half_idx = cip_U.shape[0] // 2
        cip_half_vals.append(jnp.mean(jnp.square(cip_U[half_idx:])))
        ol_half_vals.append(jnp.mean(jnp.square(ol_U[half_idx:])))

    cip_full_vals = jnp.array(cip_full_vals)
    ol_full_vals = jnp.array(ol_full_vals)

    cip_half_vals = jnp.array(cip_half_vals)
    ol_half_vals = jnp.array(ol_half_vals)

    cip_all_U = jnp.stack(cip_all_U)   # shape: (6, T)
    ol_all_U = jnp.stack(ol_all_U)     # shape: (6, T)

    print('Full episode (mean over seeds 0 to 5)')
    print(f'CIP: {jnp.mean(cip_full_vals)} ± {jnp.std(cip_full_vals)}')
    print(f'OL: {jnp.mean(ol_full_vals)} ± {jnp.std(ol_full_vals)}')

    print()
    print('Last half (mean over seeds 0 to 5)')
    print(f'CIP: {jnp.mean(cip_half_vals)} ± {jnp.std(cip_half_vals)}')
    print(f'OL: {jnp.mean(ol_half_vals)} ± {jnp.std(ol_half_vals)}')

    cip_U_mean = jnp.mean(cip_all_U, axis=0)
    cip_U_std = jnp.std(cip_all_U, axis=0)

    ol_U_mean = jnp.mean(ol_all_U, axis=0)
    ol_U_std = jnp.std(ol_all_U, axis=0)

    x = jnp.arange(cip_U_mean.shape[0])

    fig, ax = plt.subplots(1, 1)
    ax.set_title('True Actions Applied in the Dynamics (Mean over Seeds)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Control')

    ax.plot(x, cip_U_mean, label='CIP')
    ax.fill_between(x, cip_U_mean - cip_U_std, cip_U_mean + cip_U_std, alpha=0.2)

    ax.plot(x, ol_U_mean, label='OL')
    ax.fill_between(x, ol_U_mean - ol_U_std, ol_U_mean + ol_U_std, alpha=0.2)

    ax.legend()
    fig.tight_layout()
    fig.savefig('u.png', dpi=300)