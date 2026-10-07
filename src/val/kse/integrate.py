'''
Fixed-step integration and trajectory helpers for the attractors in val.kse.attractors.
'''

from typing import TYPE_CHECKING, Callable

import jax
from jax import Array

if TYPE_CHECKING:
    from val.kse.attractors import Attractor


def make_rk4(f: Callable, dt: float) -> Callable:
    '''
    Takes in a flow field and returns a discrete-time step function.
    Works on any pytree state (e.g. an augmented state like (x, h, B)) as long as f
    returns a pytree of the same structure.
    '''

    def axpy(x, k, a):
        return jax.tree.map(lambda xi, ki: xi + a * ki, x, k)

    def step(x):
        k1 = f(x)
        k2 = f(axpy(x, k1, dt / 2))
        k3 = f(axpy(x, k2, dt / 2))
        k4 = f(axpy(x, k3, dt))
        return jax.tree.map(
            lambda xi, a, b, c, d: xi + (a + 2 * b + 2 * c + d) * dt / 6,
            x, k1, k2, k3, k4)

    return step


def make_warm_up(attractor: 'Attractor') -> Callable:
    '''
    Makes the function that warms up a state so it is on the attractor.
    '''

    step = attractor.step
    burn_in = attractor.burn_in

    def scan_fn(x: Array, _):
        return step(x), None

    def warm_up(x: Array) -> Array:
        x_warm, _ = jax.lax.scan(scan_fn, x, length = burn_in)
        return x_warm

    return warm_up


def make_generate_trajectory(attractor: 'Attractor', steps: int) -> Callable:
    '''
    Builds a function which produces a trajectory from the attractor given an initial state.
    Automatically applies the specified burn_in steps so that the trajectory is likely to be on the attractor.
    '''

    step = attractor.step
    warm_up = make_warm_up(attractor)

    def scan_fn(x: Array, _):
        x = step(x)
        return x, x

    def generate_trajectory(x: Array) -> Array:
        x_warm = warm_up(x)
        _, X = jax.lax.scan(scan_fn, x_warm, length = steps)
        return X

    return generate_trajectory
