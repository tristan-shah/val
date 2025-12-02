from pathlib import Path
from typing import Callable

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from val import Dynamics, unroll_policy
from val.utils import estimate_lyapunov_hist

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state

def compute_controlled_LE(dyn: Dynamics, xt: Array, pi: callable, pi_x: callable, T: int):
    X, U = unroll_policy(dyn, xt, pi, T)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    K = jax.vmap(pi_x)(X[:-1])
    D = A + einsum(B, K, 't x1 u, t u x2 -> t x1 x2')
    lce = estimate_lyapunov_hist(D, dyn.mjx_model.opt.timestep)
    return lce

compute_controlled_LE = jax.jit(compute_controlled_LE, static_argnums = (0, 2, 3, 4))

if __name__ == '__main__':
    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    episode_len = 100
    dt = 0.05

    path = Path('checkpoints/vec_ddpg').resolve()
    manager = CheckpointManager(path)

    critic = build_pendulum_critic(3, 1, 128, rngs = nnx.Rngs(0))
    policy = build_pendulum_policy(3, 1, 128, rngs = nnx.Rngs(0))

    restored = manager.restore(39900,
        args = ocp.args.Composite(
            critic_state = ocp.args.StandardRestore(nnx.state(critic)),
            policy_state = ocp.args.StandardRestore(nnx.state(policy))
        )
    )

    nnx.update(critic, restored['critic_state'])
    nnx.update(policy, restored['policy_state'])

    ## policy and policy gradient
    pi = lambda _x: policy(normalize_pendulum_state(_x))
    pi_x = jax.jacfwd(pi)
    
    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)

    # Create a grid over theta and theta_dot
    n_theta = 200
    n_theta_dot = 200
    theta_min, theta_max = -2 * jnp.pi, 2 * jnp.pi
    theta_dot_min, theta_dot_max = -2 * jnp.pi, 2 * jnp.pi

    # Create meshgrid
    theta_grid = jnp.linspace(theta_min, theta_max, n_theta)
    theta_dot_grid = jnp.linspace(theta_dot_min, theta_dot_max, n_theta_dot)
    Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)


    lce = jax.vmap(compute_controlled_LE, in_axes = (None, 0, None, None, None))(dyn, grid_points, pi, pi_x, episode_len)
    lce = jnp.reshape(lce, (n_theta_dot, n_theta, episode_len, dyn.state_dim))  # as before

    '''
    Last LE
    '''
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Controlled Maximum FTLE, t = {episode_len * dt} (seconds)')
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')

    im = ax.imshow(
        lce[:, :, -1, 0],
        extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig('controlled_lyapunov.png', dpi = 300)
    plt.show()

    '''
    Row plot
    '''
    # # Select time steps to plot (e.g., 5 evenly spaced points)
    # num_times = 5
    # time_indices = jnp.linspace(0, episode_len-1, num_times, dtype = int)

    # fig, axes = plt.subplots(2, num_times, figsize=(4*num_times, 6))

    # for i, t in enumerate(time_indices):
    #     # First row: LE 0
    #     ax = axes[0, i]
    #     im = ax.imshow(
    #         lce[:, :, t, 0],
    #         extent=[-3*jnp.pi, 3*jnp.pi, -8, 8],
    #         origin='lower',
    #         cmap='viridis',
    #         aspect='auto'
    #     )
    #     ax.set_title(f'LE 0, t={t * dt} (s)')
    #     if i == 0:
    #         ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    #     fig.colorbar(im, ax=ax)

    #     # Second row: LE 1
    #     ax = axes[1, i]
    #     im = ax.imshow(
    #         lce[:, :, t, 1], 
    #         extent=[-3*jnp.pi, 3*jnp.pi, -8, 8],
    #         origin='lower',
    #         cmap='viridis',
    #         aspect='auto'
    #     )
    #     ax.set_title(f'LE 1, t={t * dt} (s)')
    #     if i == 0:
    #         ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    #     ax.set_xlabel(r'$\theta$ (rad)')
    #     fig.colorbar(im, ax=ax)

    # fig.tight_layout()
    # fig.savefig('controlled_lyapunov.png', dpi = 300)
    # plt.show()



    # fig, axes = plt.subplots(1, num_times, figsize=(4*num_times, 3))

    # for i, t in enumerate(time_indices):
    #     # First row: LE 0
    #     ax = axes[i]
    #     im = ax.imshow(
    #         lce[:, :, t, 0] + lce[:, :, t, 1],
    #         extent=[-3*jnp.pi, 3*jnp.pi, -8, 8],
    #         origin='lower',
    #         cmap='viridis',
    #         aspect='auto'
    #     )
    #     ax.set_title(f'Sum of LE, t={t * dt} (s)')
    #     if i == 0:
    #         ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    #     fig.colorbar(im, ax=ax)

    #     ax.set_xlabel(r'$\theta$ (rad)')

    # fig.tight_layout()
    # fig.savefig('controlled_lyapunov_sum.png', dpi = 300)
    # plt.show()