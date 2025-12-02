import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, unroll
from val.utils import estimate_lyapunov_hist

def compute_uncontrolled_LE(dyn: Dynamics, xt: Array, U: Array):
    X = unroll(dyn, xt, U)
    A, _ = jax.vmap(dyn.linearize)(X[:-1], U)
    lce = estimate_lyapunov_hist(A, dyn.mjx_model.opt.timestep)
    return lce

compute_uncontrolled_LE = jax.jit(compute_uncontrolled_LE, static_argnums = 0)

if __name__ == '__main__':
    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    episode_len = 100
    
    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dt = 0.05
    dyn = Dynamics(path = xml_path, dt = dt)

    U = jnp.zeros((episode_len, dyn.control_dim))

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


    lce = jax.vmap(compute_uncontrolled_LE, in_axes = (None, 0, None))(dyn, grid_points, U)
    lce = jnp.reshape(lce, (n_theta_dot, n_theta, episode_len, dyn.state_dim))  # as before

    '''
    Last LE
    '''
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Uncontrolled Maximum FTLE, t = {episode_len * dt} (seconds)')
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    # ax.set_aspect('equal')

    im = ax.imshow(
        lce[:, :, -1, 0],
        extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig('uncontrolled_lyapunov.png', dpi = 300)
    plt.show()


    '''
    Row plot
    '''
    # # Select time steps to plot (e.g., 5 evenly spaced points)
    # num_times = 5
    # time_indices = jnp.linspace(0, episode_len-1, num_times, dtype=int)

    # fig, axes = plt.subplots(2, num_times, figsize=(4*num_times, 6))

    # for i, t in enumerate(time_indices):
    #     # First row: LE 0
    #     ax = axes[0, i]
    #     im = ax.imshow(
    #         lce[:, :, t, 0], 
    #         extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
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
    #         extent=[theta_min, theta_max, theta_dot_min, theta_dot_max],
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
    # fig.savefig('lyapunov.png', dpi = 300)
    # plt.show()

    '''
    Sum of LE (will be zero because uncontrolled single pendulum has symmetric LE)
    '''
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
    # fig.savefig('uncontrolled_lyapunov.png', dpi = 300)
    # plt.show()