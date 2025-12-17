from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.pendulum import init_pendulum_state, normalize_pendulum_state

from main import DDPG

if __name__ == '__main__':
    seed = 105
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)

    episode_len = 400
    dt = 0.05
    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    step = make_step(dyn)
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) ## get the shape of the normalized state
    ctrl_dim = dyn.control_dim

    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 1
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    gamma = 0.99

    path = Path(f'results/checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty=0.0-hess_penalty=0.0').resolve()
    manager = ocp.CheckpointManager(path)
    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)
    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(3.1)

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