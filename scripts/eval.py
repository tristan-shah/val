from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from val import Dynamics

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state, pendulum_reward

if __name__ == '__main__':
    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    episode_len = 400
    dt = 0.05

    path = Path('checkpoints/num_env=100_vec_ddpg').resolve()
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

    @jax.jit
    def V(_x: Array):
        z = normalize_pendulum_state(_x)
        return -critic(jnp.concatenate([z, policy(z)], axis = -1)).squeeze()

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(0.0)

    X = jnp.zeros((episode_len + 1, dyn.state_dim))
    X = X.at[0].set(xt)
    v_hist = jnp.zeros((episode_len, 1))

    for t in range(episode_len):

        zt = normalize_pendulum_state(xt)
        ut = policy(zt)

        vt = V(xt)
        v_hist = v_hist.at[t].set(vt)

        xt = dyn.step(xt, ut)
        print(t, xt, ut, vt)
        X = X.at[t+1].set(xt)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-3 * jnp.pi, 3 * jnp.pi)
    ax.set_ylim(-8, 8)
    ax.plot(X[:, 0], X[:, 1])
    plt.show()

    dyn.render(X, path = 'test.mp4', skip = 1)

    plt.plot(v_hist)
    plt.show()