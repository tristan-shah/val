import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step

def make_unroll(step: callable):

    def unroll(x0: Array, U: Array):

        def body_fn(x: Array, u: Array):
            x_next = step(x, u)
            return x_next, x_next
        
        _, X = jax.lax.scan(body_fn, x0, U)
        X = jnp.concatenate([x0[None, :], X], axis = 0)

        return X
    
    return jax.jit(unroll)

TOL = 1e-6

@jax.jit
def compute_power(water_line: Array, eigs: Array):
    '''
    original
    '''
    return jnp.clip(water_line - 1 / eigs, min = 0.0)

@jax.jit
def waterfilling_solver(noise_levels: Array, total_power: float):
    '''
    Code courtesy of Noam Smilovich. 
    '''
    clipped_noise = jnp.maximum(noise_levels, TOL)
    inverse = 1.0 / clipped_noise

    inverse_sorted = jnp.sort(inverse)
    n = len(inverse_sorted)
    P = jnp.zeros_like(inverse_sorted)

    def compute_P(carry, i):
        P_prev = carry
        P_next = P_prev + i * (inverse_sorted[i] - inverse_sorted[i-1])
        return P_next, P_next

    _, P = jax.lax.scan(compute_P, 0.0, jnp.arange(1, n))
    P = jnp.concatenate([jnp.array([0.0]), P])  # P[0] = 0

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
    Code courtesy of Noam Smilovich. 
    '''
    safe_noise = jnp.maximum(noise_levels, TOL)

    def f(mu):
        return jnp.sum(compute_power(mu, safe_noise)) - total_power
    
    initial_guess = 0.0
    
    def solve(f, initial_guess):
        return waterfilling_solver(noise_levels, total_power)

    def tangent_solve(g, y):
        return y/g(1.0)
    
    return jax.lax.custom_root(f, initial_guess, solve, tangent_solve)

def compute_empoweremnt(G: Array, P: float):
    S = G.T @ G
    h2 = jnp.linalg.eigvalsh(S).clip(min = 1e-12)
    v = waterfilling_implicit(h2, P)
    p = compute_power(v, h2)
    e = 0.5 * jnp.sum(jnp.log(1 + p * h2))
    return e

if __name__ == '__main__':

    compute_empoweremnt_grad = jax.jacrev(compute_empoweremnt)

    horizon = 200
    P = 1.0

    dyn = Dynamics('xml/pendulum.xml', dt = 0.01)
    dx, du = dyn.state_dim, dyn.control_dim

    ## helper functions
    step = make_step(dyn)
    unroll = make_unroll(step)
    batch_linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    batch_quadraticize = jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(step, argnums = (0, 1) ), argnums = (0, 1) )))

    ## initial state
    xt = jnp.zeros(dx)
    xt = xt.at[0].set(3.14)

    ## initial control sequence
    U = jnp.zeros((horizon, du))

    ## unroll the trajectory
    X = unroll(xt, U)
    # dyn.render(X, 'test.mp4')

    ## compute derivatives along trajectory
    fx, fu = batch_linearize(X[:-1], U)
    (fxx, fxu), (fux, fuu) = batch_quadraticize(X[:-1], U)

    ## importance of each component of state and control
    cxx = jnp.eye(dx)
    cuu = jnp.eye(du)

    I = jnp.eye(dx)
    
    ## holding sum channel
    F = jnp.eye(dx)
    S = jnp.zeros((dx, dx, dx))
    
    ## components for calculating hessian of value
    L = cxx
    M = jnp.zeros((dx, dx))
    ## gradient of value w.r.t state
    Vx = jnp.zeros(dx)

    e_hist = jnp.zeros(horizon)
    e_grad_hist = jnp.zeros((horizon, dx, dx))

    for t in reversed(range(horizon)):
        
        ## compute hessian
        Vxx = L + M
        
        ## control to sum of states channel
        G = F @ fu[t]
        ## compute empoweremnt in this channel
        e = compute_empoweremnt(G, P)
        ## gradient of empowerment w.r.t channel matrix
        de_dG = compute_empoweremnt_grad(G, P)
        dG_dx = einsum(fu[t], S, fx[t], 'x1 u1, x x1 x2, x2 x3 -> x u1 x3') + einsum(F, fux[t], 'x x1, x1 u1 x2 -> x u1 x2')
        de_dx = einsum(de_dG, dG_dx, 'x u, x u x1 -> x1')

        print(de_dx)

        ## hessian w.r.t control
        Quu_inv = jnp.linalg.inv(cuu + fu[t].T @ Vxx @ fu[t])
        ## compute feedback gain
        K = - Quu_inv @ fu[t].T @ Vxx @ fx[t]

        ## first order closed loop derivative
        D = fx[t] + fu[t] @ K
        ## second order closed loop (total) derivative
        H = fxx[t] \
            + einsum(fxu[t], K, 'x x1 u, u x2 -> x x1 x2') \
            + einsum(K, fux[t], 'u x1, x u x2 -> x x1 x2') \
            + einsum(K, fuu[t], K, 'u1 x1, x u1 u2, u2 x2 -> x x1 x2')
        
        ## step backward
        L = D.T @ L @ D + cxx
        M = D.T @ M @ D + K.T @ cuu @ K
        Vx = Vx @ D + de_dx

        ## step backwards the gradient of the sum channel
        S = einsum(D, S, D, 'x1 x3, x x1 x2, x2 x4 -> x x3 x4') + einsum(F, H, 'x1 x2, x2 x3 x4 -> x1 x3 x4')
        ## step backward the sum channel
        F = (F + I) @ D
       
        ## save history
        e_hist = e_hist.at[t].set(e)
        e_grad_hist = e_grad_hist.at[t].set(Vx)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(e_hist)
    ax[1].plot(e_grad_hist[:, 0])
    ax[1].plot(e_grad_hist[:, 1])
    plt.show()