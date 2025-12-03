from typing import Callable, List

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import optax
import chex

def build_hidden_layers(hidden_dim: int, num_layers: int, activation: Callable, rngs: nnx.Rngs) -> List[nnx.Module]:
    layers = []

    for _ in range(num_layers):
        layers.append(nnx.Linear(hidden_dim, hidden_dim, rngs = rngs))
        layers.append(activation)

    return layers

class Critic(nnx.Module):
    def __init__(
            self, 
            rngs: nnx.Rngs,
            state_dim: int, 
            ctrl_dim: int, 
            hidden_dim: int, 
            num_layers:int, 
            activation: Callable,
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
    
    def __call__(self, x: Array, u: Array):
        return self.layers(jnp.concatenate([x, u], axis = -1))
    
class Policy(nnx.Module):
    def __init__(
            self, 
            rngs: nnx.Rngs,
            state_dim: int, 
            ctrl_dim: int, 
            hidden_dim: int, 
            num_layers:int, 
            activation: Callable
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

    def __call__(self, x: Array):
        return self.layers(x)

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

        self.normalize_state = normalize_state
        self.batch_normalize_state = jax.vmap(normalize_state)

        self.critic = Critic(rngs, state_dim, ctrl_dim, hidden_dim, num_layers_critic, activation)
        self.policy = Policy(rngs, state_dim, ctrl_dim, hidden_dim, num_layers_policy, activation)
        self.target_critic = nnx.clone(self.critic)
        self.target_policy = nnx.clone(self.policy)

        self.critic_opt = nnx.Optimizer(self.critic, optax.adam(critic_lr))
        self.policy_opt = nnx.Optimizer(self.policy, optax.adam(policy_lr))

        self.gamma = gamma

    def act(self, x: Array):
        return self.policy(self.normalize_state(x))
    
    def value(self, x: Array):
        z = self.normalize_state(x)
        return self.critic(z, self.policy(z)).squeeze()
    
    def target_value(self, x: Array):
        z = self.normalize_state(x)
        return self.target_critic(z, self.target_policy(z)).squeeze()
    
    def batch_act(self, x: Array):
        z = self.batch_normalize_state(x)
        return self.policy(z)

# @nnx.jit
# def update_critic(ddpg: DDPG, batch: tuple):

#     ## extract experience
#     x, u, r, t, x_ = batch

#     ## normalize states
#     z = ddpg.batch_normalize_state(x)
#     z_ = ddpg.batch_normalize_state(x_)

#     ## compute action in next state
#     u_ = ddpg.target_policy(z_)
#     v_ = ddpg.target_critic(z_, u_)
#     ## compute Bellman TD target
#     y = r + (1.0 - t) * ddpg.gamma * v_

#     def loss_fn(critic: Critic):
#         ## compute values
#         v = critic(z, u)
#         return jnp.mean(optax.huber_loss(v, y))
    
#     loss, grads = nnx.value_and_grad(loss_fn)(ddpg.critic)
#     ddpg.critic_opt.update(grads)
#     return loss


def make_update_critic(step: Callable, cost: Callable, grad_penalty: float):

    batch_step = jax.jit(jax.vmap(step))
    batch_linearize_step = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    batch_linearize_cost = jax.jit(jax.vmap(jax.jacfwd(cost, argnums = (0, 1))))

    @nnx.jit
    def update_critic(ddpg: DDPG, batch: tuple):

        ## extract experience
        x, u, r, t, x_next = batch

        ## construct a value function from the critic and policy
        def V(x_i: Array, C: Critic, P: Policy):
            z_i = ddpg.normalize_state(x_i)
            return C(z_i, P(z_i)).squeeze()
        
        ## differentiate the value function and broadcast over batches
        batch_Vx = jax.vmap(jax.jacrev(V), in_axes = (0, None, None))

        ## calculates the action in an individual state
        def pi(x_i: Array, P: Policy):
            return P(ddpg.normalize_state(x_i))
        
        ## broadcast policy over batches
        batch_pi = jax.vmap(pi, in_axes = (0, None))
        ## differentiate the policy and broadcast over batches
        batch_pi_x = jax.vmap(jax.jacfwd(pi), in_axes = (0, None))

        ## compute the target to regress Vx upon
        def compute_Vx_targ(x: Array):
            ## action under the current policy
            u_policy = batch_pi(x, ddpg.policy)
            K_policy = batch_pi_x(x, ddpg.policy)
            x_next_policy = batch_step(x, u_policy)

            fx, fu = batch_linearize_step(x, u_policy)
            cx, cu = batch_linearize_cost(x, u_policy)
            cx, cu = -cx, -cu ## flip sign on gradients
            D = fx + einsum(fu, K_policy, 't x1 u, t u x2 -> t x1 x2')

            Vx_next = batch_Vx(x_next_policy, ddpg.target_critic, ddpg.target_policy)

            Vx_target = cx + einsum(cu, K_policy, 'b u, b u x -> b x') + (1.0 - t) * ddpg.gamma * einsum(Vx_next, D, 'b x, b x x1 -> b x1')
            return Vx_target
        
        Vx_target = compute_Vx_targ(x)

        ## normalize states
        z = ddpg.batch_normalize_state(x)
        z_next = ddpg.batch_normalize_state(x_next)

        ## compute action in next state
        u_next = ddpg.target_policy(z_next)
        v_next = ddpg.target_critic(z_next, u_next)
        ## compute Bellman TD target
        y = r + (1.0 - t) * ddpg.gamma * v_next

        def loss_fn(critic: Critic):
            ## compute values
            v = critic(z, u)
            vx = batch_Vx(x, critic, ddpg.policy)
            
            value_loss = jnp.mean(optax.huber_loss(v, y))
            value_grad_loss = jnp.mean(optax.huber_loss(vx, Vx_target))

            loss = value_loss + value_grad_loss * grad_penalty
            aux = (value_loss, value_grad_loss)
            
            return loss, aux
        
        (loss, (value_loss, value_grad_loss)), grads = nnx.value_and_grad(loss_fn, has_aux = True)(ddpg.critic)

        ddpg.critic_opt.update(grads)
        return value_loss, value_grad_loss
    
    return update_critic

@nnx.jit
def update_policy(ddpg: DDPG, batch: tuple):

    ## extract experience
    x, _, _, _, _ = batch
    ## normalize states
    z = ddpg.batch_normalize_state(x)

    def loss_fn(policy: Policy):
        return - jnp.mean(ddpg.critic(z, policy(z)))
    
    loss, grads = nnx.value_and_grad(loss_fn)(ddpg.policy)
    ddpg.policy_opt.update(grads)
    return loss

@nnx.jit
def soft_update(model: nnx.Module, target_model: nnx.Module, tau: float = 0.005):

    new_state = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        nnx.state(model),
        nnx.state(target_model)
    )

    nnx.update(target_model, new_state)
    return target_model

@chex.dataclass(frozen = True)
class SART:
    s: chex.Array
    a: chex.Array
    r: chex.Array
    t: chex.Array