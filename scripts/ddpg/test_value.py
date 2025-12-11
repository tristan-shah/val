from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
from orbax import checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll_policy
from val.pendulum import init_pendulum_state, normalize_pendulum_state, pendulum_cost
from val.utils import make_compute_value
from val.ddpg import DDPG

from finetune import Value

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)

    gamma = 0.90
    episode_len = 400
    dt = 0.05
    dyn = Dynamics(path = 'xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    batch_init_state = jax.vmap(init_pendulum_state)
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) ## get the shape of the normalized state
    ctrl_dim = dyn.control_dim

    ## load in pretrained ddpg
    manager = ocp.CheckpointManager(
        directory = Path(f'checkpoints/hidden_dim={128}-num_layers_critic={4}-grad_penalty=0.0-hess_penalty=0.0').resolve()
    )
    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, 128, 4, 1, nnx.gelu, 1e-3, 1e-4, gamma)
    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)
    policy = ddpg.policy

    ## instantiate value function
    value_hidden_dim = 128
    value_num_layers = 5
    value_activation = nnx.gelu
    value = Value(rngs, input_dim, hidden_dim = value_hidden_dim, num_layers = value_num_layers, activation = value_activation)

    # path = Path('old_offline/train_steps=100000-value_grad-grad_penalty=0.05-value_hidden_dim=256-value_num_layers=10').resolve()
    path = Path('offline/train_steps=100000-grad_penalty=0.0-hess_penalty=0.0-value_hidden_dim=128-value_num_layers=5').resolve()
    manager = ocp.CheckpointManager(path.resolve())
    value_state = manager.restore(99500, args = ocp.args.StandardRestore(nnx.state(value)))
    nnx.update(value, value_state)


    '''
    line plot (variable theta)
    '''
    pi = jax.jit(lambda _x: policy(normalize_pendulum_state(_x)))

    ## build functions
    unroll_policy = make_unroll_policy(step, pi, episode_len)
    compute_value = make_compute_value(pendulum_cost)

    @jax.jit
    def compute_true_value(xt: Array):
        X, U = unroll_policy(xt)
        return compute_value(X, U, gamma)
    
    n_theta = 10000
    theta_dot = 2.0
    theta_min, theta_max = -2 * jnp.pi, 2.0 * jnp.pi

    # Create meshgrid
    theta_grid = jnp.linspace(theta_min, theta_max, n_theta)

    grid_points = jnp.stack([
        theta_grid,
        jnp.repeat(theta_dot, n_theta)
    ], axis = -1)

    z = jax.vmap(normalize_pendulum_state)(grid_points)

    V_true = jax.vmap(compute_true_value)(grid_points)[:, 0]
    # V_learned = jax.vmap(value)(z)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Value Function Landscape for ' + r'$\dot\theta = $' + f'{theta_dot} (rad/s)')
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel('Value')

    ax.plot(theta_grid, V_true, label = 'True Value')
    # ax.plot(theta_grid, V_learned, label = 'Learned Value')

    ax.legend()
    fig.savefig('test.png', dpi = 300)
    plt.show()

