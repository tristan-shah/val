from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val.pendulum import normalize_pendulum_state
from ddpg.main import DDPG

if __name__ == '__main__':
    seed = 105
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)
    episode_len = 400
    dt = 0.05

    state_dim = 3
    ctrl_dim = 1
    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 2
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    gamma = 0.99

    path = Path('checkpoints/grad_TEST_penalty=0.01').resolve()
    manager = ocp.CheckpointManager(path)
    
    ddpg = DDPG(rngs, normalize_pendulum_state, state_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)

    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)

    def V(_x: Array):
        z = ddpg.normalize_state(_x)
        return -ddpg.critic(z, ddpg.policy(z)).squeeze()
    
    V = jax.jit(V)

    # Create a grid over theta and theta_dot
    n_theta = 200
    n_theta_dot = 200

    # Create meshgrid
    theta_grid = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, n_theta)
    theta_dot_grid = jnp.linspace(-8, 8, n_theta_dot)
    Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis = 1)

    ## evaluate value function
    v = jax.vmap(V)(grid_points)

    ## Reshape Q-values to grid
    v_grid = v.reshape(n_theta_dot, n_theta)

    fig, ax = plt.subplots(1, 1, figsize = (8, 7))

    im = ax.imshow(
        v_grid,
        # laplacian_grid, 
        extent = [-3 * jnp.pi, 3 * jnp.pi, -8, 8],
        origin = 'lower',
        cmap = 'viridis',
        aspect = 'auto' 
    )

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