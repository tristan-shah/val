from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state

if __name__ == '__main__':

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

    def V(_x: Array):
        z = normalize_pendulum_state(_x)
        return -critic(jnp.concatenate([z, policy(z)], axis = -1)).squeeze()
    
    Vx = jax.jacrev(V)

    V = jax.jit(V)
    Vx = jax.jit(Vx)

    # Create a grid over theta and theta_dot
    n_theta = 100
    n_theta_dot = 100

    # Create meshgrid
    theta_grid = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, n_theta)
    theta_dot_grid = jnp.linspace(-8, 8, n_theta_dot)
    Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

    ## evaluate value function
    v = jax.vmap(V)(grid_points)
    # vx = jax.vmap(Vx)(grid_points)

    ## Reshape Q-values to grid
    v_grid = v.reshape(n_theta_dot, n_theta)
    # vx_grid = vx.reshape(n_theta_dot, n_theta, 2)

    fig, ax = plt.subplots(1, 1, figsize = (8, 7))

    im = ax.imshow(
        v_grid,
        # laplacian_grid, 
        extent = [-3 * jnp.pi, 3 * jnp.pi, -8, 8],
        origin = 'lower',
        cmap = 'viridis',
        aspect = 'auto' 
    )

    # step = 4  # thinning factor for arrows
    # scale = 0.3
    # # vec = vx_grid[::step, ::step]
    # vec = vx_grid
    # norm = jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8
    # vec_unit = vec / norm
    # ax.quiver(Theta[::step,::step], Theta_dot[::step,::step],
    #         -scale * vec_unit[...,0], - scale * vec_unit[...,1],
    #         color='white', alpha = 0.8, scale = 20)
    
    # ax.quiver(Theta, Theta_dot,
    #     -scale * vec_unit[...,0], - scale * vec_unit[...,1],
    #     color='white', alpha = 0.8, scale = 20)


    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    ax.set_title('Learned Value for Pendulum-v1')

    # Add colorbar (placed nicely on the side)
    cbar = fig.colorbar(im, ax = ax, shrink = 0.8, pad = 0.05)
    cbar.ax.set_ylabel('Value', rotation = 270, labelpad = 15)

    fig.tight_layout()

    # Save as square high-res image
    fig.savefig('value.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
    plt.show()