from typing import Callable
from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import flashbax as fbx
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.pendulum import init_pendulum_state, normalize_pendulum_state, pendulum_cost
from val.ddpg import SART, DDPG, Policy, build_hidden_layers, soft_update

class Value(nnx.Module):
    def __init__(
            self, 
            rngs: nnx.Rngs,
            input_dim: int, 
            hidden_dim: int, 
            num_layers:int, 
            activation: Callable,
        ):
        
        layers = []
        ## input layer
        layers.append(nnx.Linear(input_dim, hidden_dim, rngs = rngs))
        layers.append(activation)

        ## hidden layers
        layers.extend(build_hidden_layers(hidden_dim, num_layers, activation, rngs))

        ## output layers
        layers.append(nnx.Linear(hidden_dim, 1, rngs = rngs))

        self.layers = nnx.Sequential(*layers)
    
    def __call__(self, x: Array):
        return self.layers(x)
    
def make_update_value(policy: Policy, step: Callable, cost: Callable, normalize: Callable, gamma: float):

    ## normalize state before input to policy
    # pi = lambda _x: policy(normalize(_x))
    # batch_pi = jax.jit(jax.vmap(pi))
    # batch_step = jax.jit(jax.vmap(step))
    # batch_cost = jax.jit(jax.vmap(cost))
    batch_normalize = jax.jit(jax.vmap(normalize))

    @nnx.jit
    def update_value(value: Value, target_value: Value, value_opt: nnx.Optimizer, batch: tuple):

        x, u, c, t, x_next = batch

        z = batch_normalize(x)
        z_next = batch_normalize(x_next)

        v_ = jax.lax.stop_gradient(target_value(z_next))
        y = c + (1.0 - t) * gamma * v_

        def loss_fn(value):
            v = value(z)
            return jnp.mean(optax.huber_loss(v, y))
        
        loss, grad = nnx.value_and_grad(loss_fn)(value)
        value_opt.update(grad)
        return loss
    
    return update_value

if __name__ == '__main__':
    seed = 0
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)

    episode_len = 400
    dt = 0.05
    ## load in xml
    xml_path = 'xml/pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = dt)
    step = make_step(dyn)
    batch_init_state = jax.vmap(init_pendulum_state)
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) ## get the shape of the normalized state
    ctrl_dim = dyn.control_dim

    ## hyperparameters of ddpg
    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 1
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    gamma = 0.99
    tau = 0.005

    ## buffer parameters
    iterations = 100
    num_env = 1000
    episode_len = 400
    buffer_len = iterations * num_env * episode_len
    print(f'Buffer Length = {buffer_len}')

    ## can change this sampling batch size
    batch_size = 1024

    ## instantiating the buffer
    buffer = fbx.make_flat_buffer(
        max_length = buffer_len,
        min_length = batch_size,
        sample_batch_size = batch_size,
        add_batch_size = num_env
    )

    ## jit compiling buffer functions
    buffer = buffer.replace(
        init = jax.jit(buffer.init),
        add = jax.jit(buffer.add, donate_argnums = 0),
        sample = jax.jit(buffer.sample),
        can_sample = jax.jit(buffer.can_sample),
    )

    ## create dummy transition
    sart = SART(
        s = jnp.zeros(dyn.state_dim), 
        a = jnp.zeros(dyn.control_dim),
        r = jnp.zeros(1),
        t = jnp.zeros(1, dtype = bool)
    )

    ## initialize replay buffer with dummy transition
    buffer_state = buffer.init(sart)

    # === RESTORE THE BUFFER ===
    save_dir = Path('buffers/pendulum_ddpg_dataset').resolve()
    checkpointer = ocp.CheckpointManager(save_dir)
    buffer_state = checkpointer.restore(0, args = ocp.args.StandardRestore(buffer_state))

    ## load in pretrained ddpg
    path = Path(f'checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty=0.0-hess_penalty=0.0').resolve()
    manager = ocp.CheckpointManager(path)
    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)
    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)
    policy = ddpg.policy

    ## instantiate value function
    value = Value(rngs, input_dim, hidden_dim = 256, num_layers = 10, activation = nnx.gelu)
    target_value = nnx.clone(value)
    value_opt = nnx.Optimizer(value, optax.adam(1e-3))
    ## make the value update function
    update_value = make_update_value(policy, step, pendulum_cost, normalize_pendulum_state, gamma)

    loss_hist = []
    for i in range(50000):
        key, subkey = jax.random.split(key)

        data = buffer.sample(buffer_state, subkey)

        ## extract experience
        batch = (
            data.experience.first.s,
            data.experience.first.a,
            data.experience.first.r,
            data.experience.first.t,
            data.experience.second.s
        )
        
        loss = update_value(value, target_value, value_opt, batch)
        target_value = soft_update(value, target_value, tau)
        print(i, loss)
        loss_hist.append(loss)

    fig, ax = plt.subplots(1, 1)
    ax.plot(loss_hist)
    fig.savefig('loss.png', dpi = 300)

    '''
    line plot (variable theta)
    '''
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
    V_learned = jax.vmap(value)(z)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Value Function Landscape for ' + r'$\dot\theta = $' + f'{theta_dot} (rad/s)')
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel('Value')
    ax.plot(theta_grid, V_learned, label = 'Learned Value')

    ax.legend()
    fig.savefig('test.png', dpi = 300)
    plt.show()

