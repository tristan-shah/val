from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll_policy

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state
from val.pendulum import normalize_pendulum_state, pendulum_cost
from val.utils import make_compute_value, make_compute_value_grad, make_compute_value_taylor, make_compute_ddp_hessian, make_compute_ilqr_hessian

if __name__ == '__main__':
    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    episode_len = 50 #400
    dt = 0.05
    gamma = 0.99

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
    def cost_to_go(xt: Array):
        X, U = unroll_policy(xt)
        return compute_value(X, U, gamma)

    xt = jnp.zeros(dyn.state_dim)

    # xt = xt.at[0].set(-1.78)
    # xt = xt.at[1].set(2.3)
    # theta_min, theta_max = -1.784, -1.774

    ## looks nice. not representative.
    # xt = xt.at[0].set(-1.571)
    # xt = xt.at[1].set(0.1)
    # theta_min, theta_max = -1.58, -1.566

    ## negative curvature
    # xt = xt.at[0].set(0.795)
    # xt = xt.at[0].set(0.812)
    # xt = xt.at[0].set(1.31)
    xt = xt.at[0].set(1.0)
    xt = xt.at[1].set(2.0)
    theta_min, theta_max = -2.0, 2.0

    X, U = unroll_policy(xt)
    V_bar = compute_value(X, U, gamma)
    Vx, Vxx = compute_value_taylor(X, U, gamma)
    Vxx_ddp = compute_ddp_hessian(X, U, gamma)
    Vxx_ilqr = compute_ilqr_hessian(X, U, gamma)

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

    delta_x = (grid_points - xt)

    V_approx = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    V_approx_ddp = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx_ddp[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    V_approx_ilqr = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx_ilqr[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    V = jax.vmap(cost_to_go)(grid_points)[:, 0]

    fig, ax = plt.subplots(1, 1)
    ax.plot(theta_grid, jnp.log(jnp.abs(V_approx - V)), label = 'True Hessian')
    ax.plot(theta_grid, jnp.log(jnp.abs(V_approx_ddp - V)), label = 'DDP Hessian')
    ax.plot(theta_grid, jnp.log(jnp.abs(V_approx_ilqr - V)), label = 'iLQR Hessian')
    ax.legend()
    plt.show()


    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Value Function Landscape for ' + r'$\dot\theta = $' + f'{xt[1]} (rad/s)')
    ax.set_xlabel(r'$\theta$ (rad)')

    ax.set_ylabel('Value')
    # ax.set_xlim(theta_min, theta_max)
    ax.set_ylim(V.min(), V.max())

    ax.scatter(xt[0], V_bar[0], c = 'black', label = 'Expansion Point', s = 10)
    ax.plot(theta_grid, V, label = 'Value')
    ax.plot(theta_grid, V_approx, label = 'True Hessian', color = 'red')
    ax.plot(theta_grid, V_approx_ddp, label = 'DDP Hessian', color = 'orange')
    ax.plot(theta_grid, V_approx_ilqr, label = 'iLQR Hessian', color = 'purple')

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

    # V = jax.vmap(cost_to_go)(grid_points)[:, 0]
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


    # '''
    # 3d plot
    # '''
    # import jax
    # import jax.numpy as jnp
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

    # n_theta = 100
    # n_theta_dot = 100

    # theta_min, theta_max = -0.49 * jnp.pi, -0.45 * jnp.pi
    # theta_dot_min, theta_dot_max = -0.49 * jnp.pi, -0.40 * jnp.pi

    # # Create meshgrid
    # theta_grid = jnp.linspace(theta_min, theta_max, n_theta)
    # theta_dot_grid = jnp.linspace(theta_dot_min, theta_dot_max, n_theta_dot)
    # Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    # grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

    # # Compute cost-to-go
    # G = jax.vmap(compute_cost_to_go, in_axes=(None, 0, None, None, None))(dyn, grid_points, pi, episode_len, gamma)
    # G = G.reshape(n_theta_dot, n_theta)

    # # Create 3D figure
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection = '3d')

    # # Plot surface
    # surf = ax.plot_surface(
    #     Theta, Theta_dot, G,
    #     cmap='viridis',
    #     edgecolor='none',
    #     rstride = 1,
    #     cstride = 1
    # )

    # # Labels and title
    # ax.set_xlabel(r'$\theta$ (rad)')
    # ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    # ax.set_zlabel('Value')
    # ax.set_title('Explicit Value for Pendulum-v1')

    # # Add colorbar
    # fig.colorbar(surf, shrink=0.7, aspect=15, label='Value')

    # # Optional: adjust viewing angle
    # ax.view_init(elev = 30, azim = 45)

    # plt.tight_layout()
    # plt.show()
