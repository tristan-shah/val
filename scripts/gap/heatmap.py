import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

@jax.jit
def compute_vol(fx: Array, fu: Array):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du) * 1.0

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t = carry
        fx_t, fu_t = inputs

        Y_t = Q + fx_t.T @ Y_t @ fx_t
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t) @ fu_t.T @ V_t @ fx_t

        carry = (Y_t, V_t)

        return carry, carry
    
    _, (Y, V) = jax.lax.scan(scan_fn, init = (Q, Q), xs = (fx, fu), reverse = True)
    return Y, V

def make_compute_entropy(step: callable):
    
    traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step)

    def compute_entropy(x0: Array, U: Array):

        X = unroll(x0, U)
        fx, fu = traj_linerize(X[:-1], U)
        Y, V = compute_vol(fx, fu)

        ol_entropy = jnp.linalg.slogdet(Y).logabsdet
        cl_entropy = jnp.linalg.slogdet(V).logabsdet

        return ol_entropy, cl_entropy
    
    return jax.jit(compute_entropy)


if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    dt = 0.05
    horizon = 100
    eps = 0.1

    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    # dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]

    compute_entropy = make_compute_entropy(make_step(dyn))
    batch_compute_entropy = jax.jit(jax.vmap(compute_entropy, in_axes = (0, None)))

    U = jnp.zeros((horizon, dyn.control_dim)) + jax.random.normal(key, (horizon, dyn.control_dim)) * 0.05
    U = U.clip(low, high)
    T = jnp.arange(1, horizon+1)

    thetas = [0.0, 0.5, 1.0, 2.0, 3.0, 3.14]
    # thetas = [0.0, 0.5, 1.0, 2.0, 3.0, jnp.pi]

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(r'$\frac{1}{T-t}(\ln\det Y_t - \ln\det V_t)$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate of Chaos Consumption (bits)')

    for theta in thetas:
        x0 = jnp.array([theta, 0.0])
        # x0 = jnp.array([theta, 0.0, 0.0, 0.0])
        ol, cl = compute_entropy(x0, U)

        ax.plot(T * dt, (ol - cl) / jnp.flip(T*dt), label = f'{theta}')
        # ax.plot(T * dt, (ol - cl), label = f'{theta}')
        # ax.plot(T * dt, ol, label = f'{theta}')

    ax.legend(title = 'Initial Angle (rad)')
    fig.savefig('rate.png', dpi = 300)
    plt.show()




    # step = make_step(dyn)
    # traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    # unroll = make_unroll(step)

    # x0 = jnp.array([3.14, 0.0])
    # X = unroll(x0, U)
    # fx, fu = traj_linerize(X[:-1], U)












    # # Create a grid over theta and theta_dot
    # n_theta = 200
    # n_theta_dot = 200
    # theta_min, theta_max = -2 * jnp.pi, 2 * jnp.pi
    # theta_dot_min, theta_dot_max = -2 * jnp.pi, 2 * jnp.pi

    # # Create meshgrid
    # theta_grid = jnp.linspace(theta_min, theta_max, n_theta)
    # theta_dot_grid = jnp.linspace(theta_dot_min, theta_dot_max, n_theta_dot)
    # Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    # grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

    # ol, cl = batch_compute_entropy(grid_points, U)

    # ## normalize by time (make it a rate)
    # ol = ol / jnp.flip(T * dt)
    # cl = cl / jnp.flip(T * dt)

    # ol = jnp.reshape(ol, (n_theta_dot, n_theta, horizon))
    # cl = jnp.reshape(cl, (n_theta_dot, n_theta, horizon))





    # '''
    # Row plot
    # '''
    # ## Select time steps to plot (e.g., 5 evenly spaced points)
    # num_times = 4
    # time_indices = jnp.linspace(0, horizon-1, num_times, dtype = int)

    # # fig, axes = plt.subplots(2, num_times, figsize=(4*num_times, 6))
    # fig, axes = plt.subplots(3, num_times, figsize=(4*num_times, 9))
    # # fig.suptitle('Partial Derivative Controlled FLTE')

    # for i, t in enumerate(time_indices):
    #     ax = axes[0, i]
    #     im = ax.imshow(
    #         ol[:, :, t],
    #         extent = [theta_min, theta_max, theta_dot_min, theta_dot_max],
    #         origin = 'lower',
    #         cmap = 'viridis',
    #         aspect = 'auto'
    #     )
    #     ax.set_title(f'h(f), t={round(t * dt, 3)} (s)')
    #     if i == 0:
    #         ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    #     fig.colorbar(im, ax=ax)

    #     # Second row: LE 1
    #     ax = axes[1, i]
    #     im = ax.imshow(
    #         cl[:, :, t],
    #         extent = [theta_min, theta_max, theta_dot_min, theta_dot_max],
    #         origin = 'lower',
    #         cmap = 'viridis',
    #         aspect = 'auto'
    #     )
    #     ax.set_title(f'h(f | pi), t={round(t * dt, 3)} (s)')
    #     if i == 0:
    #         ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    #     ax.set_xlabel(r'$\theta$ (rad)')
    #     fig.colorbar(im, ax=ax)

    #     # Second row: LE 1
    #     ax = axes[2, i]
    #     im = ax.imshow(
    #         ol[:, :, t] - cl[:, :, t],
    #         extent = [theta_min, theta_max, theta_dot_min, theta_dot_max],
    #         origin = 'lower',
    #         cmap = 'viridis',
    #         aspect = 'auto'
    #     )
    #     ax.set_title(f'h(f) - h(f | pi), t={round(t * dt, 3)} (s)')
    #     if i == 0:
    #         ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')
    #     ax.set_xlabel(r'$\theta$ (rad)')
    #     fig.colorbar(im, ax=ax)

    # fig.tight_layout()
    # fig.savefig('entropy.png', dpi = 300)
    # plt.show()






    # '''
    # Single
    # '''
    # fig, ax = plt.subplots(1, 1)

    # im = ax.imshow(
    #     jnp.log(jnp.sum(ol - cl, axis = 2)),
    #     extent = [theta_min, theta_max, theta_dot_min, theta_dot_max],
    #     origin='lower',
    #     cmap='viridis',
    #     aspect='auto'
    # )

    # fig.colorbar(im, ax=ax)
    # ax.set_xlabel(r'$\theta$ (rad)')
    # ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')

    # fig.tight_layout()
    # fig.savefig('chaotic_controlled_lyapunov_sum.png', dpi = 300)
    # plt.show()