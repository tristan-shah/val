from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll_policy

from train import build_pendulum_critic, build_pendulum_policy, normalize_pendulum_state
from val.pendulum import normalize_pendulum_state, pendulum_cost
from val.utils import make_compute_value, make_compute_value_grad, make_compute_value_taylor, make_compute_ddp_hessian, make_compute_ilqr_hessian

if __name__ == '__main__':
    ## hyperparameters
    seed = 105
    key = jax.random.PRNGKey(seed)
    episode_len = 50 #400
    dt = 0.05
    gamma = 0.99

    path = Path('checkpoints/vec_ddpg').resolve()
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

    ## make a policy
    pi = jax.jit(lambda _x: policy(normalize_pendulum_state(_x)))

    ## load in dynamics
    dyn = Dynamics(path = 'xml/pendulum.xml', dt = dt)
    step = make_step(dyn)

    ## build functions
    unroll_policy = make_unroll_policy(step, pi, episode_len)
    ## taylor expansion
    compute_value = make_compute_value(pendulum_cost)
    compute_value_grad = make_compute_value_grad(step, pi, pendulum_cost)
    compute_value_taylor = make_compute_value_taylor(step, pi, pendulum_cost)
    compute_ddp_hessian = make_compute_ddp_hessian(step, pi, pendulum_cost)
    compute_ilqr_hessian = make_compute_ilqr_hessian(step, pi, pendulum_cost)
    
    @jax.jit
    def cost_to_go(xt: Array):
        X, U = unroll_policy(xt)
        return compute_value(X, U, gamma)

    xt_bar = jnp.zeros(dyn.state_dim)
    xt_bar = xt_bar.at[0].set(1.0)
    xt_bar = xt_bar.at[1].set(2.0)

    num_points = 1000
    key = jax.random.PRNGKey(0)
    eps = jax.random.uniform(key, shape = (num_points, dyn.state_dim)) * 2 - 1.0

    delta_x = eps * 0.01
    xt = xt_bar + delta_x

    X, U = unroll_policy(xt_bar)
    V_bar = compute_value(X, U, gamma)
    Vx, Vxx = compute_value_taylor(X, U, gamma)
    Vxx_ddp = compute_ddp_hessian(X, U, gamma)
    Vxx_ilqr = compute_ilqr_hessian(X, U, gamma)

    V_approx = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    V_approx_ddp = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx_ddp[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    V_approx_ilqr = V_bar[0] + delta_x @ Vx[0] + 0.5 * einsum(delta_x, Vxx_ilqr[0], delta_x, 'b x1, x1 x2, b x2 -> b')
    V = jax.vmap(cost_to_go)(xt)[:, 0]

    delta_x_norm = jnp.linalg.norm(delta_x, axis = -1)
    r_true = jnp.abs(V - V_approx)
    r_ddp = jnp.abs(V - V_approx_ddp)
    r_ilqr = jnp.abs(V - V_approx_ilqr)

    print(delta_x_norm)