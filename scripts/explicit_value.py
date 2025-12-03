from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll_policy

from val.pendulum import normalize_pendulum_state, pendulum_cost
from val.utils import make_compute_value, make_compute_value_grad, make_compute_value_taylor, make_compute_ddp_hessian, make_compute_ilqr_hessian
from val.ddpg import DDPG

if __name__ == '__main__':
    seed = 105
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)
    episode_len = 400
    dt = 0.05

    state_dim = 3
    ctrl_dim = 1
    hidden_dim = 128
    num_layers_critic = 8 #4
    num_layers_policy = 2
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    gamma = 0.99

    path = Path('checkpoints/num_layers_critic=8-grad_penalty=0.01').resolve()
    manager = ocp.CheckpointManager(path)

    ddpg = DDPG(rngs, normalize_pendulum_state, state_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)

    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)

    critic = ddpg.critic
    policy = ddpg.policy

    ## make a policy
    pi = jax.jit(lambda _x: policy(normalize_pendulum_state(_x)))

    ## load in dynamics
    dyn = Dynamics(path = 'xml/pendulum.xml', dt = dt)
    step = make_step(dyn)

    ## build functions
    unroll_policy = make_unroll_policy(step, pi, episode_len)
    ## taylor expansion
    compute_value = make_compute_value(pendulum_cost)
    compute_value_grad = make_compute_value_grad(step, pi, pendulum_cost)
    compute_value_taylor = make_compute_value_taylor(step, pi, pendulum_cost)
    compute_ddp_hessian = make_compute_ddp_hessian(step, pi, pendulum_cost)
    compute_ilqr_hessian = make_compute_ilqr_hessian(step, pi, pendulum_cost)
    
    @jax.jit
    def explicit_value(xt: Array):
        X, U = unroll_policy(xt)
        return compute_value(X, U, gamma)
    
    @jax.jit
    def learned_value(xt: Array):
        zt = normalize_pendulum_state(xt)
        return -critic(zt, policy(zt)).squeeze()
    
    ## select a state
    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(1.0)
    xt = xt.at[1].set(-2.0)
    theta_min, theta_max = -2.0, 2.0

    # ## unroll a nominal
    # X, U = unroll_policy(xt)
    # ## compute taylor expansions
    # V_bar = compute_value(X, U, gamma)
    # Vx, Vxx = compute_value_taylor(X, U, gamma)
    # Vxx_ddp = compute_ddp_hessian(X, U, gamma)
    # Vxx_ilqr = compute_ilqr_hessian(X, U, gamma)

    '''
    line plot (variable theta)
    '''
    n_theta = 10000
    theta_dot = xt[1]

    # Create meshgrid
    theta_grid = jnp.linspace(theta_min, theta_max, n_theta)

    grid_points = jnp.stack([
        theta_grid,
        jnp.repeat(theta_dot, n_theta)
    ], axis = -1)


    ## evaluate true function over grid points
    V = jax.vmap(explicit_value)(grid_points)[:, 0]
    V_learned = jax.vmap(learned_value)(grid_points)

    ## evaluate second order taylor approximations
    # delta_x = (grid_points - xt)
    # V_approx = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    # V_approx_ddp = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx_ddp[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    # V_approx_ilqr = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx_ilqr[0], delta_x, 'b x1, x1 x2, b x2 -> b')


    '''
    Error plot
    '''
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(theta_grid, jnp.log(jnp.abs(V_approx - V)), label = 'True Hessian')
    # ax.plot(theta_grid, jnp.log(jnp.abs(V_approx_ddp - V)), label = 'DDP Hessian')
    # ax.plot(theta_grid, jnp.log(jnp.abs(V_approx_ilqr - V)), label = 'iLQR Hessian')
    # ax.legend()
    # plt.show()


    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Value Function Landscape for ' + r'$\dot\theta = $' + f'{xt[1]} (rad/s)')
    ax.set_xlabel(r'$\theta$ (rad)')

    ax.set_ylabel('Value')
    # ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(V.min(), V.max())
    
    ax.plot(theta_grid, V, label = 'Value')
    ax.plot(theta_grid, V_learned, label = 'Learned Value')
    # ax.scatter(xt[0], V_bar[0], c = 'black', label = 'Expansion Point', s = 10)
    # ax.plot(theta_grid, V_approx, label = 'True Hessian', color = 'red')
    # ax.plot(theta_grid, V_approx_ddp, label = 'DDP Hessian', color = 'orange')
    # ax.plot(theta_grid, V_approx_ilqr, label = 'iLQR Hessian', color = 'purple')

    ax.legend()
    # fig.savefig(f'theta={xt[0].item()}_theta-dot={xt[1].item()}.png', dpi = 300)
    fig.savefig('test.png', dpi = 300)
    plt.show()



    '''
    heatmap
    '''
    # theta_min, theta_max = -1.52, -1.5
    # theta_dot_min, theta_dot_max = -1.52, -1.5

    # theta_min, theta_max = -2 * jnp.pi, 2 * jnp.pi
    # theta_dot_min, theta_dot_max = -2 * jnp.pi, 2 * jnp.pi

    # n_theta = 200
    # n_theta_dot = 200
    # # Create meshgrid
    # theta_grid = jnp.linspace(theta_min, theta_max, n_theta)
    # theta_dot_grid = jnp.linspace(theta_dot_min, theta_dot_max, n_theta_dot)
    # Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    # grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

    # V = jax.vmap(explicit_value)(grid_points)[:, 0]
    # V = V.reshape(n_theta_dot, n_theta)

    # fig, ax = plt.subplots(1, 1, figsize = (8, 7))  # Adjust size as needed (7,7) works well for 300 DPI

    # im = ax.imshow(
    #     V,
    #     extent = [theta_min, theta_max, theta_dot_min, theta_dot_max],
    #     origin = 'lower',
    #     cmap = 'viridis',
    #     aspect = 'auto'  # Let imshow handle aspect; we'll enforce squareness via figure size
    # )

    # ax.set_xlabel(r'$\theta$ (rad)')
    # ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    # ax.set_title('Explicit Value for Pendulum-v1')

    # # Add colorbar (placed nicely on the side)
    # cbar = fig.colorbar(im, ax = ax, shrink = 0.8, pad = 0.05)
    # cbar.ax.set_ylabel('Value', rotation = 270, labelpad = 15)

    # # Optional: tighten layout but prevent colorbar from being cropped
    # fig.tight_layout()

    # # Save as square high-res image
    # fig.savefig('explicity_value.png', dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
    # plt.show()