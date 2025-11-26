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

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state, pendulum_reward

@jax.jit
def C(_x: Array, _u: Array):
    return -pendulum_reward(_x, _u).squeeze()

linearize_cost = jax.jacobian(C, argnums = (0, 1))
quadraticize_cost = jax.jacfwd(linearize_cost, argnums = (0, 1))

def compute_discounted_cost(X: Array, U:Array, gamma: float):
    c = jax.vmap(C)(X[:-1], U)

    def body_fun(G: Array, c_t: Array):
        G = c_t + gamma * G
        return G, G
    
    G0, _ = jax.lax.scan(body_fun, init = 0.0, xs = c, reverse = True)
    return G0

compute_discounted_cost = jax.jit(compute_discounted_cost)

def compute_cost_to_go(dyn: Dynamics, xt: Array, pi: callable, T: int, gamma: float):
    X, U = unroll_policy(dyn, xt, pi, T)
    return compute_discounted_cost(X, U, gamma)

compute_cost_to_go = jax.jit(compute_cost_to_go, static_argnums = (0, 2, 3))

def compute_cost_grad(dyn: Dynamics, xt: Array, pi: callable, pi_x: callable, T: int, gamma: float):
    
    X, U = unroll_policy(dyn, xt, pi, T)
    fx, fu = jax.vmap(dyn.linearize)(X[:-1], U)
    K = jax.vmap(pi_x)(X[:-1])
    D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')

    cx, cu = jax.vmap(linearize_cost)(X[:-1], U)

    def body_fun(Gx: Array, inputs: tuple):
        D_t, K_t, cx_t, cu_t = inputs
        Gx = gamma * D_t.T @ Gx + cx_t + K_t.T @ cu_t
        return Gx, Gx

    Gx_T = jnp.zeros_like(xt)
    _, Gx_t = jax.lax.scan(body_fun, init = Gx_T, xs = (D, K, cx, cu), reverse = True)

    return jnp.concatenate([Gx_t, Gx_T[None, :]], axis = 0)

compute_cost_grad = jax.jit(compute_cost_grad, static_argnums = (0, 2, 3, 4))

if __name__ == '__main__':
    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    episode_len = 400
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

    ## policy and policy gradient
    pi = jax.jit(lambda _x: policy(normalize_pendulum_state(_x)))
    pi_x = jax.jit(jax.jacfwd(pi))
    pi_xx = jax.jit(jax.jacfwd(pi_x))

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    quadraticize = jax.jacfwd(dyn.linearize, argnums = (0, 1))

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(0.1)
    xt = xt.at[1].set(0.0)

    X, U = unroll_policy(dyn, xt, pi, episode_len)
    fx, fu = jax.vmap(dyn.linearize)(X[:-1], U)
    (fxx, fxu), (fux, fuu) = jax.vmap(quadraticize)(X[:-1], U)

    K = jax.vmap(pi_x)(X[:-1])
    KK = jax.vmap(pi_xx)(X[:-1])

    D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
    cx, cu = jax.vmap(linearize_cost)(X[:-1], U)
    (cxx, cxu), (cux, cuu) = jax.vmap(quadraticize_cost)(X[:-1], U)

    Czz = cxx \
        + einsum(cxu, K, 't x1 u, t u x2 -> t x1 x2') \
        + einsum(K, cux, 't u x1, t u x2 -> t x1 x2') \
        + einsum(K, cuu, K, 't u1 x1, t u1 u2, t u2 x2 -> t x1 x2')

    W = Czz + einsum(cu, KK, 't u, t u x1 x2 -> t x1 x2')

    H = fxx \
        + einsum(fxu, K, 't x x1 u, t u x2 -> t x x1 x2') \
        + einsum(K, fux, 't u x1, t x u x2 -> t x x1 x2') \
        + einsum(K, fuu, K, 't u1 x1, t x u1 u2, t u2 x2 -> t x x1 x2') \
        + einsum(fu, KK, 't x u, t u x1 x2 -> t x x1 x2')

    Vx = compute_cost_grad(dyn, xt, pi, pi_x, episode_len, gamma)
    term = einsum(Vx[1:], H, 't x, t x x1 x2 -> t x1 x2')

    Vxx = jnp.zeros((dyn.state_dim, dyn.state_dim))
    Vxx_hist = [jnp.trace(Vxx).item()]

    for t in reversed(range(episode_len)):
        Vxx = gamma * D[t].T @ Vxx @ D[t] + gamma * term[t] + W[t]
        Vxx_hist.insert(0, jnp.trace(Vxx).item())
        print(jnp.trace(Vxx))

    # print(Vxx_hist)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(Vxx_hist)
    # plt.show()

    print(Vxx)

    # scalar value at t=0 (must use same pi as unroll_policy)
    V0_fun = lambda x: compute_cost_to_go(dyn, x, pi, episode_len, gamma)

    # autodiff / exact Hessian from JAX
    # Vxx_fd = jax.hessian(V0_fun)(xt)   # or jax.jacfwd(jax.grad(V0_fun))(xt)
    Vx_fd = jax.jacfwd(V0_fun)
    Vxx_fd = jax.jacfwd(Vx_fd)
    print(Vxx_fd(xt))






    # n_theta = 100
    # n_theta_dot = 100
    # # Create meshgrid
    # theta_grid = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, n_theta)
    # theta_dot_grid = jnp.linspace(-8, 8, n_theta_dot)
    # Theta, Theta_dot = jnp.meshgrid(theta_grid, theta_dot_grid)
    # grid_points = jnp.stack([Theta.ravel(), Theta_dot.ravel()], axis=1)

    # G = jax.vmap(compute_cost_to_go, in_axes = (None, 0, None, None, None))(dyn, grid_points, pi, episode_len, gamma)
    # G = G.reshape(n_theta_dot, n_theta)

    # Gx = jax.vmap(compute_cost_grad, in_axes = (None, 0, None, None, None, None))(dyn, grid_points, pi, pi_x, episode_len, gamma)
    # Gx = Gx.reshape(n_theta_dot, n_theta, dyn.state_dim)

    # fig, ax = plt.subplots(1, 1, figsize = (8, 7))  # Adjust size as needed (7,7) works well for 300 DPI

    # im = ax.imshow(
    #     G,
    #     extent = [-3 * jnp.pi, 3 * jnp.pi, -8, 8],
    #     origin = 'lower',
    #     cmap = 'viridis',
    #     aspect = 'auto'  # Let imshow handle aspect; we'll enforce squareness via figure size
    # )

    # # step =   # thinning factor for arrows
    # scale = 0.3
    # # vec = Gx[::step, ::step]
    # vec = Gx
    # norm = jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8
    # vec_unit = vec / norm
    # # ax.quiver(Theta[::step,::step], Theta_dot[::step,::step],
    # #         -scale * vec_unit[...,0], - scale * vec_unit[...,1],
    # #         color='white', alpha = 0.8, scale = 20)
    # ax.quiver(Theta, Theta_dot,
    #     -scale * vec_unit[...,0], - scale * vec_unit[...,1],
    #     color = 'white', alpha = 0.8, scale = 20)

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