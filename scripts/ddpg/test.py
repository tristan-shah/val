from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics
from val.pendulum import normalize_pendulum_state

from main import DDPG

if __name__ == '__main__':
    seed = 105
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)
    episode_len = 400
    dt = 0.05

    state_dim = 3
    ctrl_dim = 1
    hidden_dim = 128
    num_layers_critic = 8
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

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    xt = jnp.zeros(dyn.state_dim)

    X = jnp.zeros((episode_len + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    for t in range(episode_len):

        ut = ddpg.act(xt)
        xt = dyn.step(xt, ut)
        print(t, xt, ut)
        X = X.at[t+1].set(xt)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-3 * jnp.pi, 3 * jnp.pi)
    ax.set_ylim(-8, 8)
    ax.plot(X[:, 0], X[:, 1])
    plt.show()

    dyn.render(X, path = 'test.mp4', skip = 1)