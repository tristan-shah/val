import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

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
    return 0.5 * jnp.sum(jnp.log(1.0 + p * h2))

class iEmpowerment:
    def __init__(self, dyn: Dynamics):

        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        self.dx = dyn.state_dim
        self.du = dyn.control_dim
        ## helper functions
        self.step = make_step(dyn)
        self.unroll = make_unroll(self.step)
        self.batch_linearize = jax.jit(jax.vmap(jax.jacfwd(self.step, argnums = (0, 1))))
        self.batch_quadraticize = jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(self.step, argnums = (0, 1) ), argnums = (0, 1) )))

    def backward(self, X: Array, U: Array, e: Array):

        ## compute derivatives along trajectory
        fx, fu = self.batch_linearize(X[:-1], U)
        (fxx, fxu), (fux, fuu) = self.batch_quadraticize(X[:-1], U)

        dx, du = self.dx, self.du
        ## importance of each component of state and control
        cxx = jnp.eye(dx)
        cuu = jnp.eye(du)

        ## holding sum channel
        F = jnp.eye(dx)
        ## gradient of the sum channel
        S = jnp.zeros((dx, dx, dx))

        ## hessian and gradient of value function
        Vxx = cxx
        # Vx = e[-1] 
        Vx = jnp.zeros(dx)

        horizon = U.shape[0]
        K = jnp.zeros((horizon, du, dx))
        k = jnp.zeros((horizon, du))

        for t in reversed(range(horizon)):
            
            ## control to sum of states channel
            G = F @ fu[t]
            ## gradient of empowerment w.r.t channel matrix
            de_dG = compute_empoweremnt_grad(G, P)
            ## gradient of the channel w.r.t state
            dG_dx = einsum(fu[t], S, fx[t], 'x1 u1, x x1 x2, x2 x3 -> x u1 x3') + einsum(F, fux[t], 'x x1, x1 u1 x2 -> x u1 x2')
            ## tensor chain rule
            de_dx = einsum(de_dG, dG_dx, 'x u, x u x1 -> x1')

            ## hessian w.r.t control
            Quu_inv = jnp.linalg.inv(cuu + fu[t].T @ Vxx @ fu[t])
            ## compute feedback gain

            K = K.at[t].set(
                - Quu_inv @ fu[t].T @ Vxx @ fx[t]
            )

            k = k.at[t].set(
                - Quu_inv @ (fu[t].T @ Vx)
            )

            ## first order closed loop derivative
            D = fx[t] + fu[t] @ K[t]
            ## second order closed loop (total) derivative
            H = fxx[t] \
                + einsum(fxu[t], K[t], 'x x1 u, u x2 -> x x1 x2') \
                + einsum(K[t], fux[t], 'u x1, x u x2 -> x x1 x2') \
                + einsum(K[t], fuu[t], K[t], 'u1 x1, x u1 u2, u2 x2 -> x x1 x2')
            
            ## riccati equation
            Vxx = cxx + fx[t].T @ Vxx @ fx[t] - fx[t].T @ Vxx @ fu[t] @ Quu_inv @ fu[t].T @ Vxx @ fx[t]

            Vx = (-de_dx) + D.T @ Vx ## empowerment gradient propagation
            # Vx = e[t] + D.T @ Vx ## original value gradient propagation

            ## step backwards the gradient of the sum channel
            S = einsum(D, S, D, 'x1 x3, x x1 x2, x2 x4 -> x x3 x4') + einsum(F, H, 'x1 x2, x2 x3 x4 -> x1 x3 x4')
            ## step backward the sum channel
            # F = (F + jnp.eye(dx)) @ D
            F = F @ D

        return k, K
    
    def forward(self, X: Array, U: Array, k: Array, K: Array, alpha: float):

        low = self.low
        high = self.high
        T = k.shape[0]

        xt = X[0]
        X_new = jnp.zeros_like(X)
        X_new = X_new.at[0].set(xt)
        U_new = jnp.zeros_like(k)

        for t in range(T):

            ## compute deviation in current trajectory
            delta_x = X_new[t] - X[t]

            ## compute new control
            ut = U[t] + alpha * k[t] + K[t] @ delta_x
            ut = ut.clip(low, high)

            ## propagate dynamics with new control
            xt = self.step(xt, ut)

            X_new = X_new.at[t+1].set(xt)
            U_new = U_new.at[t].set(ut)

        return X_new, U_new



if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    
    compute_empoweremnt_grad = jax.jit(jax.jacrev(compute_empoweremnt))

    horizon = 400
    P = 1.0

    dyn = Dynamics('xml/pendulum.xml', dt = 0.01)
    
    dx, du = dyn.state_dim, dyn.control_dim

    iemp = iEmpowerment(dyn)

    ## initial state
    xt = jnp.zeros(dx)
    xt = xt.at[0].set(3.14)

    ## initial control sequence
    U = jnp.zeros((horizon, du)) + jax.random.normal(key, (horizon, du)) * 0.01
    U = U.clip(iemp.low, iemp.high)

    ## unroll the trajectory
    X = iemp.unroll(xt, U)
    # e = compute_pendulum_error(X)

    alpha = 10000.0

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(-1.1, 1.1)
    
    for i in range(20):

        # k, K = iemp.backward(X, U, e)
        # X, U = iemp.forward(X, U, k, K, alpha)

        print(U.T)
        ax.plot(U)

    plt.show()

    dyn.render(X, path = 'test.mp4')



    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(k)
    # ax[1].plot(K[:, 0, 0])
    # ax[1].plot(K[:, 0, 1])
    # plt.show()