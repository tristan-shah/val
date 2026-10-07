'''
Shared harness for every estimator. An estimator is an auxiliary state (h and S, Q and nu, ...)
carried alongside x; the runner warms x up on the attractor, burns in the auxiliary state along
the trajectory, then records running estimates.
'''

from dataclasses import dataclass
from typing import Any, Callable

import jax
from jax import Array
from jax import numpy as jnp

from val.kse.attractors import Attractor
from val.kse.integrate import make_warm_up


@dataclass(frozen = True)
class Estimator:
    '''
    Holds the functions that define an estimator. The state is any pytree that contains x.
    '''
    init:       Callable[[Array], Any]          ## x -> initial state
    step:       Callable[[Any], Any]            ## advances the state by one step of dt
    restart:    Callable[[Any], Any]            ## called after the auxiliary burn-in: zero the accumulators
    readout:    Callable[[Any, Array], dict]    ## (state, t) -> running estimates; always includes 'kse' (sum of positive
                                                ## exponents); 'neg' (sum of negative exponents) where the method provides it


def make_evaluate(attractor: Attractor, estimator: Estimator, steps: int, record_every: int = 1) -> Callable:
    '''
    Builds a function x -> dict of running estimates, each with leading dimension
    steps // record_every. Burn-in (for x, then for the auxiliary state) uses attractor.burn_in.
    '''

    if steps % record_every != 0:
        raise ValueError(f'steps ({steps}) must be a multiple of record_every ({record_every})')

    burn_in = attractor.burn_in
    warm_up = make_warm_up(attractor)
    T = jnp.arange(1, steps // record_every + 1) * record_every * attractor.dt

    def advance(state, length: int):
        state, _ = jax.lax.scan(lambda s, _: (estimator.step(s), None), state, length = length)
        return state

    def scan_fn(state, t: Array):
        state = estimator.step(state) if record_every == 1 else advance(state, record_every)
        return state, estimator.readout(state, t)

    def evaluate(x: Array) -> dict:

        ## warms up the starting state to the attractor, then the auxiliary state along the trajectory
        state = estimator.init(warm_up(x))
        state = estimator.restart(advance(state, burn_in))

        _, out = jax.lax.scan(scan_fn, state, T)
        return out

    return evaluate
