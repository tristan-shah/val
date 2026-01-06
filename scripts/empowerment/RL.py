from typing import Callable
from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from flax import nnx
import optax
import flashbax as fbx
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.pendulum import init_pendulum_state, normalize_pendulum_state
from val.ddpg import Critic, SART, soft_update, build_hidden_layers

class Critic(nnx.Module):
    def __init__(
            self, 
            rngs: nnx.Rngs,
            state_dim: int, 
            ctrl_dim: int, 
            hidden_dim: int, 
            num_layers:int, 
            activation: Callable,
            normalize_state: Callable
        ):
        
        layers = []
        ## input layer
        layers.append(nnx.Linear(state_dim + ctrl_dim, hidden_dim, rngs = rngs))
        layers.append(activation)

        ## hidden layers
        layers.extend(build_hidden_layers(hidden_dim, num_layers, activation, rngs))

        ## output layers
        layers.append(nnx.Linear(hidden_dim, 1, rngs = rngs))

        self.layers = nnx.Sequential(*layers)
        self.normalize_state = normalize_state
    
    def __call__(self, x: Array, u: Array):
        return self.layers(jnp.concatenate([self.normalize_state(x), u], axis = -1))

class Policy(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            state_dim: int, 
            ctrl_dim: int, 
            hidden_dim: int,
            num_layers: int, 
            activation: Callable,
            normalize_state: Callable
        ):
        
        layers = []
        ## input layer
        layers.append(nnx.Linear(state_dim, hidden_dim, rngs = rngs))
        layers.append(activation)

        ## hidden layers
        layers.extend(build_hidden_layers(hidden_dim, num_layers, activation, rngs))

        ## output layers
        layers.append(nnx.Linear(hidden_dim, ctrl_dim, rngs = rngs))
        layers.append(nnx.tanh)

        self.layers = nnx.Sequential(*layers)
        self.normalize_state = normalize_state
        self.batch_normalize_state = jax.vmap(normalize_state)

    def __call__(self, x: Array):
        # return self.layers(self.normalize_state(x))
        return self.layers(self.batch_normalize_state(x))

class DDPG(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            normalize_state: Callable,
            state_dim: int, 
            ctrl_dim: int, 
            hidden_dim: int, 
            num_layers_critic: int,
            num_layers_policy: int,
            activation: Callable,
            critic_lr: float, 
            policy_lr: float,
            gamma: float
        ):

        self.critic = Critic(rngs, state_dim, ctrl_dim, hidden_dim, num_layers_critic, activation, normalize_state)
        self.policy = Policy(rngs, state_dim, ctrl_dim, hidden_dim, num_layers_policy, activation, normalize_state)
        self.target_critic = nnx.clone(self.critic)
        self.target_policy = nnx.clone(self.policy)

        self.critic_opt = nnx.Optimizer(self.critic, optax.adam(critic_lr))
        self.policy_opt = nnx.Optimizer(self.policy, optax.adam(policy_lr))

        self.gamma = gamma

    def act(self, x: Array):
        return self.policy(x)
    
    def value(self, x: Array):
        return self.critic(x, self.policy(x))
    
    def target_value(self, x: Array):
        return self.target_critic(x, self.target_policy(x))

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)

    dt = 0.05
    num_env = 100
    episode_len = 400
    buffer_len = 1_000_000
    num_episodes = 100

    gamma = 0.99
    hidden_dim = 128
    num_layers_critic = 4
    num_layers_policy = 1
    activation = nnx.gelu
    critic_lr = 1e-3
    policy_lr = 1e-4
    batch_size = 4
    tau = 0.005
    exploration_noise = 0.1

    ## load in dynamics
    dyn = Dynamics(path = 'xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    batch_step = jax.vmap(step)
    linearize = jax.jit(jax.jacfwd(step, argnums = (0, 1)))
    batch_linearize = jax.vmap(linearize)

    low = dyn.mjx_model.actuator_ctrlrange[:, 0]
    high = dyn.mjx_model.actuator_ctrlrange[:, 1]
    input_dim = len(normalize_pendulum_state(init_pendulum_state(key))) ## get the shape of the normalized state
    ctrl_dim = dyn.control_dim

    ddpg = DDPG(rngs, normalize_pendulum_state, input_dim, ctrl_dim, hidden_dim, num_layers_critic, num_layers_policy, activation, critic_lr, policy_lr, gamma)
    policy_x = jax.jacfwd(ddpg.policy)

    ## fake batch
    x = jax.random.normal(key, (batch_size, dyn.state_dim))
    u = jax.random.normal(key, (batch_size, dyn.control_dim))
    r = jax.random.normal(key, (batch_size,))
    t = jnp.zeros((batch_size, 1))
    x_ = jax.random.normal(key, (batch_size, dyn.state_dim))
    batch = (x, u, r, t, x_)

    # print(ddpg.policy(x))
    # print(policy_x(x).shape)