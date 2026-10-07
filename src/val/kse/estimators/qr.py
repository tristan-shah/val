'''
QR baseline for Lyapunov exponents.

discrete QR (Benettin et al. 1980, Shimada & Nagashima 1979): push an orthonormal frame
through the Jacobian of the step map and re-orthonormalize with a QR factorization every step.
The sum of the positive exponents is the Pesin estimate of the KSE.
'''

from typing import Callable, NamedTuple

import jax
from jax import Array
from jax import numpy as jnp

from val.kse.attractors import Attractor
from val.kse.runner import Estimator


class QRState(NamedTuple):
    x: Array
    Q: Array    ## orthonormal frame, n x p
    nu: Array   ## accumulated log growth of each frame direction, p


def positive_sum(lyap: Array) -> Array:
    '''
    Sum of the positive exponents, i.e. the Pesin estimate of the KSE.
    '''
    return jnp.sum(jnp.maximum(lyap, 0.0), axis = -1)


def negative_sum(lyap: Array) -> Array:
    '''
    Sum of the negative exponents.
    '''
    return jnp.sum(jnp.minimum(lyap, 0.0), axis = -1)


def _readout(lyap: Array) -> dict:
    return {'kse': positive_sum(lyap), 'neg': negative_sum(lyap), 'lyap': lyap}


def _init(attractor: Attractor, p: int) -> Callable:

    Q0 = jnp.eye(attractor.dim, p)

    def init(x: Array) -> QRState:
        return QRState(x, Q0, jnp.zeros(p, dtype = x.dtype))

    return init


def make_discrete_qr(attractor: Attractor, p: int | None = None) -> Estimator:
    '''
    Tracks the top p exponents; p defaults to the full spectrum.
    '''

    p = attractor.dim if p is None else p
    step = attractor.step
    jacobian = jax.jacfwd(step)

    def qr_step(state: QRState) -> QRState:

        x, Q, nu = state

        ## push the frame through the tangent map and re-orthonormalize
        Q, R = jnp.linalg.qr(jacobian(x) @ Q)
        nu += jnp.log(jnp.abs(jnp.diag(R)))

        return QRState(step(x), Q, nu)

    def restart(state: QRState) -> QRState:
        return state._replace(nu = jnp.zeros_like(state.nu))

    def readout(state: QRState, t: Array) -> dict:
        return _readout(state.nu / t)

    return Estimator(_init(attractor, p), qr_step, restart, readout)
