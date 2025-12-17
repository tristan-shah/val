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

from val import Dynamics, make_step, make_unroll_policy
from val.utils import estimate_lyapunov_hist
from val.ddpg import DDPG, Policy
from val.pendulum import normalize_pendulum_state, init_pendulum_state

def make_compute_controlled_LE(dyn: Dynamics, pi: callable, T: int):

    step = make_step(dyn)
    traj_linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    unroll_policy = make_unroll_policy(step, pi, T)
    traj_pi_x = jax.vmap(jax.jacfwd(pi))
    dt = dyn.mjx_model.opt.timestep

    def compute_controlled_LE(xt: Array):
        X, U = unroll_policy(xt)
        A, B = traj_linearize(X[:-1], U)
        K = traj_pi_x(X[:-1])

        ## closed loop derivatives
        D = A + einsum(B, K, 't x1 u, t u x2 -> t x1 x2')

        lce = estimate_lyapunov_hist(D, dt)
        return lce
    
    return jax.jit(compute_controlled_LE)

if __name__ == '__main__':

    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)
    episode_len = 100
    dt = 0.05

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    ## get the shape of the normalized state
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) 
    ctrl_dim = dyn.control_dim

    ## load in ddpg
    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 1
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    gamma = 0.99

    path = Path(f'results/checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty=0.0-hess_penalty=0.0').resolve()
    manager = ocp.CheckpointManager(path)
    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)
    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)
    policy = ddpg.policy

    ## build policy
    pi = lambda _x: policy(normalize_pendulum_state(_x))

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

    # lce = jax.vmap(compute_controlled_LE, in_axes = (None, 0, None, None, None))(dyn, grid_points, pi, pi_x, episode_len)
    compute_controlled_LE = make_compute_controlled_LE(dyn, pi, episode_len)
    lce = jax.vmap(compute_controlled_LE)(grid_points)
    lce = jnp.reshape(lce, (n_theta_dot, n_theta, episode_len, dyn.state_dim))  # as before

    '''
    Last LE
    '''
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Controlled Sum FTLE, t = {episode_len * dt} (seconds)')
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')

    im = ax.imshow(
        lce[:, :, -1, 0] + lce[:, :, -1, 1],
        extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig('last_controlled_lyapunov.png', dpi = 300)
    plt.show()

    '''
    Row plot
    '''
    # Select time steps to plot (e.g., 5 evenly spaced points)
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
    #         extent=[-3 * jnp.pi, 3 * jnp.pi, -8, 8],
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