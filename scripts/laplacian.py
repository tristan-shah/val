from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from val import Dynamics, unroll

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
    episode_len = 2#400
    dt = 0.05
    gamma = 0.99

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


    ## build value function from critic and policy
    def V(_x: Array):
        z = normalize_pendulum_state(_x)
        return -critic(jnp.concatenate([z, policy(z)], axis = -1)).squeeze()
    
    Vx_ad = jax.jacrev(V)




    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)

    pi = lambda _x: policy(normalize_pendulum_state(_x))
    pi_x = jax.jacfwd(pi)
    pi_xx = jax.jacfwd(pi_x)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(2.5)
    xt = xt.at[1].set(-1.0)

    X, U = unroll_policy(dyn, xt, pi, episode_len)
    fx, fu = jax.vmap(dyn.linearize)(X[:-1], U)
    K = jax.vmap(pi_x)(X[:-1])
    ## total derivative
    D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
    
    compute_hessians = jax.jacfwd(dyn.linearize, argnums = (0, 1))
    (fxx, fxu), (fux, fuu) = jax.vmap(compute_hessians)(X[:-1], U)
    print(fxx.shape, fxu.shape, fux.shape, fuu.shape, K.shape)

    ## compute second order total derivative
    H = fxx \
        + einsum(fxu, K, 't x1 x2 u, t u x3 -> t x1 x2 x3') \
        + einsum(K, fux, 't u x3, t x1 x2 u -> t x1 x2 x3') \
        + einsum(K, fuu, K, 't u1 x2, t x1 u1 u2, t u2 x3 -> t x1 x2 x3')

    print(jax.vmap(pi_xx)(X[:-1]))


    # cx = jax.vmap(Cx)(X[:-1], U)
    # cu = jax.vmap(Cu)(X[:-1], U)


    # ## initial values for Vx and Vxx
    # Vx = jnp.zeros(dyn.state_dim)
    # Vxx = jnp.zeros((dyn.state_dim, dyn.state_dim))

    # ## history for tracking Vx and Vxx values backwards
    # Vx_hist = jnp.zeros((episode_len + 1, dyn.state_dim))
    # Vx_hist = Vx_hist.at[-1].set(Vx)

    # ## backwards recursion
    # for t in reversed(range(episode_len)):
    #     # Vxx = gamma * ( D[t].T @ Vxx @ D[t] + Vx.T @ )

    #     Vx = gamma * D[t].T @ Vx + cx[t] + K[t].T @ cu[t]
    #     Vx_hist = Vx_hist.at[t].set(Vx)
    #     print(t, Vx)

    # Vx_ad_hist = jax.vmap(Vx_ad)(X)

    # fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    # fig.suptitle(r'$\theta_0 = $' + f'{xt[0]}, ' + r'$\dot\theta_0 = $' + f'{xt[1]}')

    # ax[0].set_title(r'Gradient of Value w.r.t $\theta_t$')
    # ax[0].set_xlabel('Timestep: t')
    # ax[0].plot(Vx_hist[:, 0], label = 'Recursion')
    # ax[0].plot(Vx_ad_hist[:, 0], label = 'Autodiff')
    # ax[0].legend()

    # ax[1].set_title(r'Gradient of Value w.r.t $\dot\theta_t$')
    # ax[1].set_xlabel('Timestep: t')
    # ax[1].plot(Vx_hist[:, 1], label = 'Recursion')
    # ax[1].plot(Vx_ad_hist[:, 1], label = 'Autodiff')
    # ax[1].legend()
    # fig.savefig(f'theta0={xt[0]}_dottheta0={xt[1]}_value_gradient.png', dpi = 300)
    # plt.show()