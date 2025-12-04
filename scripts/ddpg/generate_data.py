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
    pi = lambda _x: policy(normalize(_x))
    batch_pi = jax.jit(jax.vmap(pi))
    batch_step = jax.jit(jax.vmap(step))
    batch_cost = jax.jit(jax.vmap(cost))
    batch_normalize = jax.jit(jax.vmap(normalize))

    @nnx.jit
    def update_value(value: Value, target_value: Value, value_opt: nnx.Optimizer, x: Array):
        
        u = batch_pi(x)
        x_next = batch_step(x, u)
        c = batch_cost(x, u)

        z = batch_normalize(x)
        z_next = batch_normalize(x_next)

        v_ = jax.lax.stop_gradient(target_value(z_next))
        y = c[:, None] + gamma * v_

        def loss_fn(value):
            v = value(z)
            return jnp.mean(optax.huber_loss(v, y))
        
        loss, grad = nnx.value_and_grad(loss_fn)(value)
        value_opt.update(grad)
        return loss
    
    return update_value

if __name__ == '__main__':
    seed = 105
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)

    ## buffer parameters
    iterations = 100
    num_env = 1000
    episode_len = 400
    buffer_len = iterations * num_env * episode_len
    batch_size = 128 ## dummy argument for sampling
    print(f'Buffer Length = {buffer_len}')

    dt = 0.05
    ## load in xml
    dyn = Dynamics(path = 'xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    batch_step = jax.jit(jax.vmap(step))
    batch_init_state = jax.jit(jax.vmap(init_pendulum_state))
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

    ## load in pretrained ddpg
    path = Path(f'checkpoints/hidden_dim={hidden_dim}-num_layers_critic={num_layers_critic}-grad_penalty=0.0-hess_penalty=0.0').resolve()
    manager = ocp.CheckpointManager(path)
    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)
    ddpg_state = manager.restore(39900, args = ocp.args.StandardRestore(nnx.state(ddpg)))
    nnx.update(ddpg, ddpg_state)
    policy = ddpg.policy
    pi = lambda _x: policy(normalize_pendulum_state(_x))
    batch_pi = jax.jit(jax.vmap(pi))
    batch_cost = jax.jit(jax.vmap(pendulum_cost))

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

    for _ in range(iterations):
        key, subkey = jax.random.split(key)
        ## initialize a random batch of states
        batch_key = jax.random.split(subkey, num_env)
        x = batch_init_state(batch_key)

        done = jnp.zeros((num_env, 1), dtype = jnp.bool)

        for i in range(episode_len):

            u = batch_pi(x)
            c = batch_cost(x, u)[:, None]
            x_next = batch_step(x, u)

            ## check if episode is done
            done = done.at[:].set(jnp.bool(i == episode_len - 1))

            ## store transition
            sart = SART(s = x,  a = u, r = c, t = done)
            buffer_state = buffer.add(buffer_state, sart)

            ## overwrite previous state
            x = x_next

        print(f'Iteration = {i}')

    # === SAVE THE BUFFER ===
    save_dir = Path('buffers/pendulum_ddpg_dataset').resolve()
    save_dir.mkdir(parents = True, exist_ok = True)

    checkpointer = ocp.CheckpointManager(save_dir)

    checkpointer.save(0, args = ocp.args.StandardSave(buffer_state))
    checkpointer.close()
    print(f'Offline dataset saved to {save_dir}')