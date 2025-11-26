from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from val import Dynamics

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state, pendulum_reward
from controlled_lyapunov_exponents import unroll_policy

@jax.jit
def C(_x: Array, _u: Array):
    return -pendulum_reward(_x, _u).squeeze()

Cx = jax.jacrev(C, argnums = 0)
Cu = jax.jacrev(C, argnums = 1)
Cxx = jax.jacrev(Cx, argnums = 0)
Cux = jax.jacrev(Cu, argnums = 0)
Cxu = jax.jacrev(Cx, argnums = 1)
Cuu = jax.jacrev(Cu, argnums = 1)

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


    ## build value function from critic and policy
    def V(_x: Array):
        _z = normalize_pendulum_state(_x)
        return -critic(jnp.concatenate([_z, policy(_z)], axis = -1)).squeeze()
    
    # Vx_ad = jax.jacrev(V)
    # Vxx_ad = jax.jacfwd(Vx_ad)
    Vx_ad = jax.jacobian(V)
    Vxx_ad = jax.hessian(V)


    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)

    pi = lambda _x: policy(normalize_pendulum_state(_x))
    pi_x = jax.jacfwd(pi)
    pi_xx = jax.jacfwd(pi_x)
    compute_hessians = jax.jacfwd(dyn.linearize, argnums = (0, 1))

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(0.9)
    xt = xt.at[1].set(0.0)

    ## unroll a trajectory under a given policy starting from xt
    X, U = unroll_policy(dyn, xt, pi, episode_len)
    
    ## linearize dynamics along nominal trajectory
    fx, fu = jax.vmap(dyn.linearize)(X[:-1], U)
    ## extract policy gradient
    K = jax.vmap(pi_x)(X[:-1])
    ## total derivative
    D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
    
    ## compute second order derivatives of the dynamics
    (fxx, fxu), (fux, fuu) = jax.vmap(compute_hessians)(X[:-1], U)
    ## extract policy hessian
    KK = jax.vmap(pi_xx)(X[:-1])
    ## compute second order total derivative
    H = fxx \
        + einsum(fxu, K, 't x1 x2 u, t u x3 -> t x1 x2 x3') \
        + einsum(K, fux, 't u x3, t x1 x2 u -> t x1 x2 x3') \
        + einsum(K, fuu, K, 't u1 x2, t x1 u1 u2, t u2 x3 -> t x1 x2 x3') \
        + einsum(fu, KK, 't x1 u, t u x2 x3 -> t x1 x2 x3')
    
    ## first order cost derivatives
    cx = jax.vmap(Cx)(X[:-1], U)
    cu = jax.vmap(Cu)(X[:-1], U)
    ## second order cost derivatives
    cxx = jax.vmap(Cxx)(X[:-1], U)
    cxu = jax.vmap(Cxu)(X[:-1], U)
    cux = jax.vmap(Cux)(X[:-1], U)
    cuu = jax.vmap(Cuu)(X[:-1], U)

    ## last values for Vx and Vxx
    Vx = jnp.zeros(dyn.state_dim)
    Vxx = jnp.zeros((dyn.state_dim, dyn.state_dim))
    # Vx = Vx_ad(X[-1])
    # Vxx = Vxx_ad(X[-1])

    ## history for tracking Vx and Vxx values backwards
    Vx_hist = jnp.zeros((episode_len + 1, dyn.state_dim))
    Vxx_hist = jnp.zeros((episode_len + 1, dyn.state_dim, dyn.state_dim))

    ## set last values in history
    Vx_hist = Vx_hist.at[-1].set(Vx)
    Vxx_hist = Vxx_hist.at[-1].set(Vxx)

    ## backwards recursion
    for t in reversed(range(episode_len)):

        ## step value hessian backwards
        Vxx = \
            + cxx[t] \
            + einsum(cxu[t], K[t], 'x1 u, u x2 -> x1 x2') \
            + einsum(K[t], cux[t], 'u x1, u x2 -> x1 x2') \
            + einsum(K[t], cuu[t], K[t], 'u1 x1, u1 u2, u2 x2 -> x1 x2') \
            + einsum(cu[t], KK[t], 'u, u x1 x2 -> x1 x2') \
            + gamma * D[t].T @ Vxx @ D[t] \
            + gamma * einsum(Vx, H[t], 'x, x x1 x2 -> x1 x2')
        
        ## step value gradient backwards
        Vx = (cx[t] + K[t].T @ cu[t]) + gamma * D[t].T @ Vx

        ## log history
        Vx_hist = Vx_hist.at[t].set(Vx)
        Vxx_hist = Vxx_hist.at[t].set(Vxx)
        print(t)


    Vx_ad_hist = jax.vmap(Vx_ad)(X)
    Vxx_ad_hist = jax.vmap(Vxx_ad)(X)

    plt.plot(jnp.trace(Vxx_hist, axis1 = 1, axis2 = 2))
    plt.plot(jnp.trace(Vxx_ad_hist, axis1 = 1, axis2 = 2))
    plt.show()





    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    fig.suptitle(r'$\theta_0 = $' + f'{xt[0]}, ' + r'$\dot\theta_0 = $' + f'{xt[1]}')

    ax[0].set_title(r'Gradient of Value w.r.t $\theta_t$')
    ax[0].set_xlabel('Timestep: t')
    ax[0].plot(Vx_hist[:, 0], label = 'Recursion')
    ax[0].plot(Vx_ad_hist[:, 0], label = 'Autodiff')
    ax[0].legend()

    ax[1].set_title(r'Gradient of Value w.r.t $\dot\theta_t$')
    ax[1].set_xlabel('Timestep: t')
    ax[1].plot(Vx_hist[:, 1], label = 'Recursion')
    ax[1].plot(Vx_ad_hist[:, 1], label = 'Autodiff')
    ax[1].legend()
    fig.savefig(f'theta0={xt[0]}_dottheta0={xt[1]}_value_gradient.png', dpi = 300)
    plt.show()