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

@nnx.jit
def update_critic(ddpg: DDPG, batch: tuple):

    ## extract experience
    x, u, r, t, x_ = batch

    ## normalize states
    z = ddpg.batch_normalize_state(x)
    z_ = ddpg.batch_normalize_state(x_)

    ## compute action in next state
    u_ = ddpg.target_policy(z_)
    v_ = ddpg.target_critic(z_, u_)
    ## compute Bellman TD target
    y = r + (1.0 - t) * ddpg.gamma * v_

    def loss_fn(critic: Critic):
        ## compute values
        v = critic(z, u)
        return jnp.mean(optax.huber_loss(v, y))
    
    loss, grads = nnx.value_and_grad(loss_fn)(ddpg.critic)
    ddpg.critic_opt.update(grads)
    return loss


def make_update_critic(step: Callable, cost: Callable, grad_penalty: float, hess_penalty: float):

    linearize_step = jax.jacfwd(step, argnums = (0, 1))
    quadraticize_step = jax.jacfwd(linearize_step, argnums = (0, 1))

    linearize_cost = jax.jacfwd(cost, argnums = (0, 1))
    quadraticize_cost = jax.jacfwd(linearize_cost, argnums = (0, 1))

    batch_step = jax.jit(jax.vmap(step))
    batch_linearize_step = jax.jit(jax.vmap(linearize_step))
    batch_quadraticize_step = jax.jit(jax.vmap(quadraticize_step))
    batch_linearize_cost = jax.jit(jax.vmap(linearize_cost))
    batch_quadraticize_cost = jax.jit(jax.vmap(quadraticize_cost))

    @nnx.jit
    def update_critic(ddpg: DDPG, batch: tuple):

        ## extract experience
        x, u, r, t, x_next = batch

        gamma = ddpg.gamma

        ## construct a value function from the critic and policy
        def V(x_i: Array, C: Critic, P: Policy):
            z_i = ddpg.normalize_state(x_i)
            return C(z_i, P(z_i)).squeeze()
        
        ## calculates the action in an individual state
        def pi(x_i: Array, P: Policy):
            return P(ddpg.normalize_state(x_i))
        
        ## differentiate the value function and broadcast over batches
        Vx = jax.jacrev(V)
        Vxx = jax.jacrev(Vx)
        batch_Vx = jax.vmap(Vx, in_axes = (0, None, None))
        batch_Vxx = jax.vmap(Vxx, in_axes = (0, None, None))

        ## differentiate the policy and broadcast over batches
        pi_x = jax.jacfwd(pi)
        pi_xx = jax.jacfwd(pi_x)
        batch_pi = jax.vmap(pi, in_axes = (0, None))
        batch_pi_x = jax.vmap(pi_x, in_axes = (0, None))
        batch_pi_xx = jax.vmap(pi_xx, in_axes = (0, None))

        def compute_targets(x: Array):
            ## action under the current policy
            u_policy = batch_pi(x, ddpg.policy)
            
            K = batch_pi_x(x, ddpg.policy)
            KK = batch_pi_xx(x, ddpg.policy)

            ## propagate dynamics under current policy
            x_next_policy = batch_step(x, u_policy)

            fx, fu = batch_linearize_step(x, u_policy)
            (fxx, fxu), (fux, fuu) = batch_quadraticize_step(x, u_policy)

            cx, cu = batch_linearize_cost(x, u_policy)
            (cxx, cxu), (cux, cuu) = batch_quadraticize_cost(x, u_policy)

            ## first order closed loop (total) derivative
            D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
            ## second order closed loop (total) derivative of dynamics
            H = fxx \
                + einsum(fxu, K, 't x x1 u, t u x2 -> t x x1 x2') \
                + einsum(K, fux, 't u x1, t x u x2 -> t x x1 x2') \
                + einsum(K, fuu, K, 't u1 x1, t x u1 u2, t u2 x2 -> t x x1 x2') \
                + einsum(fu, KK, 't x u, t u x1 x2 -> t x x1 x2')
            
            ## first derivative of instantanious cost
            cz = cx + einsum(cu, K, 't u, t u x -> t x')

            ## hessian of instantanious cost
            czz = cxx \
                + einsum(cxu, K, 't x1 u, t u x2 -> t x1 x2') \
                + einsum(K, cux, 't u x1, t u x2 -> t x1 x2') \
                + einsum(K, cuu, K, 't u1 x1, t u1 u2, t u2 x2 -> t x1 x2') \
                + einsum(cu, KK, 't u, t u x1 x2 -> t x1 x2')
            
            ## value gradient and value hessian computed in the next state
            Vx_next = batch_Vx(x_next_policy, ddpg.target_critic, ddpg.target_policy)
            Vxx_next = batch_Vxx(x_next_policy, ddpg.target_critic, ddpg.target_policy)

            ## compute the gradient target
            Vx_target = cz + (1.0 - t) * gamma * einsum(Vx_next, D, 'b x, b x x1 -> b x1')

            ## compute the hessian target
            pullback = einsum(D, Vxx_next, D, 'b x1 x2, b x1 x3, b x3 x4 -> b x2 x4')
            pushforward = einsum(Vx_next, H, 'b x, b x x1 x2 -> b x1 x2')

            Vxx_target = (1.0 - t)[:, :, None] * gamma * (pullback + pushforward) + czz

            return Vx_target, Vxx_target

        Vx_target, Vxx_target = compute_targets(x)

        ## normalize states
        z = ddpg.batch_normalize_state(x)
        z_next = ddpg.batch_normalize_state(x_next)

        ## compute action in next state
        u_next = ddpg.target_policy(z_next)
        ## compute Bellman TD target
        V_target = r + (1.0 - t) * gamma * ddpg.target_critic(z_next, u_next)

        def loss_fn(critic: Critic):
            ## compute values
            v = critic(z, u)
            vx = batch_Vx(x, critic, ddpg.policy)
            vxx = batch_Vxx(x, critic, ddpg.policy)
            
            value_loss = jnp.mean(optax.huber_loss(v, V_target))
            value_grad_loss = jnp.mean(optax.huber_loss(vx, Vx_target))
            value_hess_loss = jnp.mean(optax.huber_loss(vxx, Vxx_target))

            loss = value_loss + value_grad_loss * grad_penalty + value_hess_loss * hess_penalty
            aux = (value_loss, value_grad_loss, value_hess_loss)
            return loss, aux
        
        (loss, (value_loss, value_grad_loss, value_hess_loss)), grads = nnx.value_and_grad(loss_fn, has_aux = True)(ddpg.critic)

        ddpg.critic_opt.update(grads)
        return value_loss, value_grad_loss, value_hess_loss
    
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

    new_target_model = nnx.clone(target_model)

    new_state = jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        nnx.state(model),
        nnx.state(new_target_model)
    )

    nnx.update(new_target_model, new_state)
    return new_target_model

@chex.dataclass(frozen = True)
class SART:
    s: chex.Array
    a: chex.Array
    r: chex.Array
    t: chex.Array