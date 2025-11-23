from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from val import Dynamics, unroll

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state


if __name__ == '__main__':

    path = Path('checkpoints/ddpg').resolve()
    manager = CheckpointManager(path)

    critic = build_pendulum_critic(3, 1, 128, rngs = nnx.Rngs(0))
    policy = build_pendulum_policy(3, 1, 128, rngs = nnx.Rngs(0))

    restored = manager.restore(39860,
        args = ocp.args.Composite(
            critic_state = ocp.args.StandardRestore(nnx.state(critic)),
            policy_state = ocp.args.StandardRestore(nnx.state(policy))
        )
    )

    nnx.update(critic, restored['critic_state'])
    nnx.update(policy, restored['policy_state'])

    def V(_x: Array):
        z = normalize_pendulum_state(_x)
        return -critic(jnp.concatenate([z, policy(z)], axis = -1)).squeeze()
    
    Vx = jax.jacrev(V)
    Vxx = jax.jacrev(Vx)

    V = jax.jit(V)
    Vx = jax.jit(Vx)
    Vxx = jax.jit(Vxx)

    # Create a grid over theta and theta_dot
    n_theta = 200
    n_theta_dot = 200

    # Create meshgrid
    theta_grid = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, n_theta)
    theta_dot_grid = jnp.linspace(-8, 8, n_theta_dot)
    Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

    ## evaluate value function
    q_values = jax.vmap(V)(grid_points)

    ## evaluate laplacian of value function
    laplacian = jnp.trace(jax.vmap(Vxx)(grid_points), axis1 = 1, axis2 = 2)
    laplacian = laplacian.clip(-500, 500)

    # Reshape Q-values to grid
    q_grid = q_values.reshape(n_theta_dot, n_theta)
    laplacian_grid = laplacian.reshape(n_theta_dot, n_theta)

    fig, ax = plt.subplots(1, 1, figsize = (8, 7))  # Adjust size as needed (7,7) works well for 300 DPI

    im = ax.imshow(
        # q_grid,
        laplacian_grid, 
        extent = [-3 * jnp.pi, 3 * jnp.pi, -8, 8],
        origin = 'lower',
        cmap = 'viridis',
        aspect = 'auto'  # Let imshow handle aspect; we'll enforce squareness via figure size
    )

    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    ax.set_title('Laplacian for Pendulum-v1')

    # Add colorbar (placed nicely on the side)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    # cbar.ax.set_ylabel('Value', rotation=270, labelpad=15)
    cbar.ax.set_ylabel('Laplacian', rotation=270, labelpad=15)

    # Optional: tighten layout but prevent colorbar from being cropped
    fig.tight_layout()

    # Save as square high-res image
    fig.savefig('laplacian.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()