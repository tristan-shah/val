'''
Riccati estimators of the sum of positive Lyapunov exponents, from the growth of a noise-driven
tangent covariance P. logdet P grows like 2 * sum(max(lambda_i, 0)) * t, so the estimate is h / 2t.
P obeys a Lyapunov equation; the estimators propagate its inverse S = P^{-1}, which obeys a Riccati
equation.

discrete    P_{k+1} = A P_k A^T + I, i.e. S_{k+1} = (I + A S_k^{-1} A^T)^{-1}, with A the Jacobian of
            the step map. This is Equation (10) of the paper (Table 2).
continuous  P' = J P + P J^T + I, i.e. S' = -S J - J^T S - S^2, as an ODE for a square-root factor
            B of S = B B^T. This is the continuous-time analogue used for Figure 3.
'''

from typing import Callable, NamedTuple

import jax
from jax import Array
from jax import numpy as jnp

from val.kse.attractors import Attractor
from val.kse.integrate import make_rk4
from val.kse.runner import Estimator


class DiscreteRiccatiState(NamedTuple):
    x: Array
    h: Array        ## accumulated entropy, logdet P
    S: Array        ## information matrix, P^{-1}


class RiccatiState(NamedTuple):
    x: Array
    h: Array        ## accumulated entropy, logdet P
    B: Array        ## square-root factor of the information matrix, S = B B^T = P^{-1}
    a: Array        ## accumulated auxiliary term tr(S); -a / 2t -> sum of negative exponents


def make_discrete_riccati(attractor: Attractor) -> Estimator:

    step = attractor.step
    jacobian = jax.jacfwd(step)
    I = jnp.eye(attractor.dim)

    def init(x: Array) -> DiscreteRiccatiState:
        return DiscreteRiccatiState(x, jnp.zeros((), dtype = x.dtype), I)

    def riccati_step(state: DiscreteRiccatiState) -> DiscreteRiccatiState:

        ## unpack attractor state and information matrix
        x, h, S = state

        ## evaluate the jacobian
        A = jacobian(x)
        M = S + A.T @ A

        ## increment h and update the information matrix
        h += jnp.linalg.slogdet(M).logabsdet
        S = I - A @ jnp.linalg.inv(M) @ A.T

        return DiscreteRiccatiState(step(x), h, S)

    def restart(state: DiscreteRiccatiState) -> DiscreteRiccatiState:
        return state._replace(h = jnp.zeros_like(state.h))

    def readout(state: DiscreteRiccatiState, t: Array) -> dict:
        ## one half time normalized entropy accumulation
        return {'kse': 0.5 * state.h / t}

    return Estimator(init, riccati_step, restart, readout)


def make_riccati_ode(f: Callable) -> Callable:
    '''
    Augments the flow f with (h, B, a) so that h(t) / 2t -> sum of positive Lyapunov exponents
    and -a(t) / 2t -> sum of negative Lyapunov exponents, where a accumulates the auxiliary term tr(S).

    S = B B^T is the inverse of P, the covariance of tangent vectors driven by unit white noise:
        P' = J P + P J^T + I
    so that
        d/dt logdet P = tr(P^{-1} P') = 2 tr J + tr S             (= h_dot)
        S' = -S P' S = -S J - J^T S - S^2                         (reproduced by B_dot below)
    '''

    jacobian = jax.jacfwd(f)

    def riccati_ode(state: RiccatiState) -> RiccatiState:

        x, h, B, a = state

        J = jacobian(x)

        x_dot = f(x)
        a_dot = jnp.sum(B * B)
        h_dot = a_dot + 2 * jnp.trace(J)
        B_dot = -(0.5 * B @ B.T + J.T) @ B

        return RiccatiState(x_dot, h_dot, B_dot, a_dot)

    return riccati_ode


def make_continuous_riccati(attractor: Attractor) -> Estimator:
    '''
    Integrates make_riccati_ode with RK4 at the attractor's dt.
    '''

    I = jnp.eye(attractor.dim)

    def init(x: Array) -> RiccatiState:
        zero = jnp.zeros((), dtype = x.dtype)
        return RiccatiState(x, zero, I, zero)

    def restart(state: RiccatiState) -> RiccatiState:
        zero = jnp.zeros_like(state.h)
        return state._replace(h = zero, a = zero)

    def readout(state: RiccatiState, t: Array) -> dict:
        ## h / 2t -> sum of positive exponents, and the auxiliary integral gives the negative ones directly
        return {'kse': 0.5 * state.h / t, 'neg': -0.5 * state.a / t}

    return Estimator(init, make_rk4(make_riccati_ode(attractor.f), attractor.dt), restart, readout)
