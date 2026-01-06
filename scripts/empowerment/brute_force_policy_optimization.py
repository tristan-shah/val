from typing import Callable

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import optax
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.pendulum import normalize_pendulum_state
from val.ddpg import build_hidden_layers

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

    def __call__(self, x: Array):
        return self.layers(self.normalize_state(x))
    
def make_unroll_policy(step: Callable, horizon: int):

    def unroll_policy(xt: Array, policy: nnx.Module):

        def scan_fn(_xt: Array, _):
            _ut = policy(_xt)
            _xt = step(_xt, _ut)
            return _xt, (_xt, _ut)
        
        _, (X, U) = jax.lax.scan(scan_fn, init = xt, length = horizon)
        X = jnp.concatenate([xt[None, :], X], axis = 0)
        return X, U
    
    return nnx.jit(unroll_policy)

def make_update_policy(step: Callable, horizon: int):

    traj_linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll_policy = make_unroll_policy(step, horizon)

    @nnx.jit
    def update_policy(xt: Array, pi: Policy, pi_opt: nnx.Optimizer):

        graphdef = nnx.graphdef(pi)

        def loss_fn(pi_state: nnx.State):

            pi = nnx.merge(graphdef, pi_state)

            ## roll out a sequence of states and controls
            X, U = unroll_policy(xt, pi)
            ## linearize the dynamics along the trajectory
            fx, fu = traj_linearize(X[:-1], U)
            ## compute the gradient of the policy
            # K = jax.vmap(jax.jacfwd(pi))(X[:-1])
            ## compute closed-loop derivative
            # D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')


            ## maximum eigenvalue
            # ln_abs_det_ol = jnp.log(jnp.linalg.svd(fx, compute_uv = False).max(axis = -1))

            ## difference in determinants
            # _, ln_abs_det_ol = jnp.linalg.slogdet(fx)
            # _, ln_abs_det_cl = jnp.linalg.slogdet(D)

            ## compute the difference between open and closed-loop FTLE
            # loss = jnp.mean(ln_abs_det_ol - ln_abs_det_cl)

            S = einsum(fx, fx, 't x x1, t x x2 -> t x1 x2')
            ln_abs_det_ol = jnp.trace(S, axis1 = 1, axis2 = 2)

            loss = jnp.mean(ln_abs_det_ol)
            aux = (loss, X)
            ## for minimization
            return -loss, aux

        pi_state = nnx.state(pi)
        grads, aux = jax.jacfwd(loss_fn, has_aux = True)(pi_state)
        pi_opt.update(grads)

        return aux

    return update_policy

if __name__ == '__main__':

    seed = 0
    rngs = nnx.Rngs(seed)
    hidden_dim = 128
    num_layers = 2
    activation = nnx.gelu

    ## environment parameters
    dt = 0.05
    horizon = 100

    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    step = make_step(dyn)
    traj_linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))

    state_dim = len(normalize_pendulum_state(jnp.zeros(dyn.state_dim)))
    pi = Policy(rngs, state_dim, dyn.control_dim, hidden_dim, num_layers, activation, normalize_pendulum_state)

    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(0.0)

    unroll_policy = make_unroll_policy(step, horizon)
    X, U = unroll_policy(xt, pi)

    update_policy = make_update_policy(step, horizon)
    pi_opt = nnx.Optimizer(pi, optax.adam(1e-3))

    hist = []
    for i in range(100):
        loss, X = update_policy(xt, pi, pi_opt)
        print(i, loss, X[-1])

        hist.append(loss)

        fig, ax = plt.subplots(1, 1)
        ax.plot(hist)
        fig.tight_layout()
        fig.savefig('loss.png', dpi = 300)
        plt.close(fig)

        dyn.render(X, f'test_{i}.mp4', skip = 1)