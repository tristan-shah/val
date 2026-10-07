'''
Empowerment baseline under a linear-Gaussian channel approximation (Tiomkin et al. 2024).

The final state of a horizon-T rollout is linearized in the control sequence, F = d x_T / d U,
and the channel capacity of the Gaussian channel with gain F under a total power constraint P
is computed by water-filling over the eigenvalues of F F^T.
'''

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum

from val import Dynamics

TOL = 1e-6


@jax.jit
def compute_power(water_line: Array, eigs: Array):
    return jnp.clip(water_line - 1 / eigs, min = 0.0)


def unroll(dyn: Dynamics, xt: Array, U: Array):
    '''
    Rolls the dynamics out over U, returning (horizon + 1, state_dim) including xt.
    '''

    def body_fun(xt_: Array, ut_: Array):
        xt_next = dyn.step(xt_, ut_)
        return xt_next, xt_next

    _, X = jax.lax.scan(body_fun, xt, U)
    return jnp.concatenate([xt[None, :], X])

unroll = jax.jit(unroll, static_argnums = 0)


@jax.jit
def compute_F_from_A_B(A: Array, B: Array):
    '''
    Sensitivities of the final state to each action from the per-step jacobians (A_t, B_t).
    '''

    I = jnp.eye(A.shape[-1])

    def body_fun(_I: Array, ab: tuple):
        a, b = ab
        return _I @ a, _I @ b

    _, F = jax.lax.scan(body_fun, I, (A, B), reverse = True)

    return F


@jax.jit
def waterfilling_solver(noise_levels: Array, total_power: float):
    '''
    Water level mu such that sum_i max(mu - 1 / eig_i, 0) = total_power, by bisection over
    the sorted breakpoints.
    '''
    clipped_noise = jnp.maximum(noise_levels, TOL)
    inverse = 1.0 / clipped_noise

    inverse_sorted = jnp.sort(inverse)
    n = len(inverse_sorted)

    def compute_P(carry, i):
        P_prev = carry
        P_next = P_prev + i * (inverse_sorted[i] - inverse_sorted[i-1])
        return P_next, P_next

    _, P = jax.lax.scan(compute_P, 0.0, jnp.arange(1, n))
    P = jnp.concatenate([jnp.array([0.0]), P])

    def cond_fun(state):
        bot, top = state
        return top - bot > 1

    def body_fun(state):
        bot, top = state
        mid = (bot + top) // 2
        new_bot = jax.lax.cond(total_power >= P[mid], lambda: mid, lambda: bot)
        new_top = jax.lax.cond(total_power >= P[mid], lambda: top, lambda: mid)
        return new_bot, new_top

    bot, _ = jax.lax.while_loop(cond_fun, body_fun, (0, n))

    mu = inverse_sorted[bot] + (total_power - P[bot]) / (bot + 1)
    return mu


@jax.jit
def waterfilling_implicit(noise_levels: Array, total_power: float):
    '''
    Water-filling solution wrapped in `jax.lax.custom_root` so it is differentiable.
    '''
    safe_noise = jnp.maximum(noise_levels, TOL)

    def f(mu):
        return jnp.sum(compute_power(mu, safe_noise)) - total_power

    def solve(f, initial_guess):
        return waterfilling_solver(noise_levels, total_power)

    def tangent_solve(g, y):
        return y / g(1.0)

    return jax.lax.custom_root(f, 0.0, solve, tangent_solve)


def compute_empowerment(dyn: Dynamics, xt: Array, U: Array, P: float):
    '''
    Empowerment (nats) of state xt over the horizon of U with total control power P.
    '''
    X = unroll(dyn, xt, U)
    A, B = jax.vmap(dyn.linearize)(X[:-1], U)
    F = compute_F_from_A_B(A, B)
    F = jnp.permute_dims(F, (1, 0, 2))

    ## covariance of the final state under unit-variance actions
    S = einsum(F, F, 'x1 T u, x2 T u -> x1 x2')
    h2 = jnp.linalg.eigvalsh(S).clip(min = 1e-12)
    v = waterfilling_implicit(h2, P)
    p = compute_power(v, h2)
    return 0.5 * jnp.sum(jnp.log(1 + p * h2))

compute_empowerment = jax.jit(compute_empowerment, static_argnums = 0)
compute_empowerment_grad = jax.jit(jax.jacfwd(compute_empowerment, argnums = 1), static_argnums = 0)
