'''
Chaotic attractors used to validate the KSE estimators (Figure 3 and Table 2 of the paper).

Each attractor is described by a frozen config dataclass, a dynamics builder and a reset builder,
collected in ATTRACTOR_REGISTRY and instantiated through build_attractor(...). Flows are stepped
with RK4 at the registry's default dt (overridable); maps are stepped directly.
'''

from dataclasses import dataclass
from typing import Any, Callable

import jax
from jax import Array
from jax import numpy as jnp

from val.kse.integrate import make_rk4


# --------------------------------------------------------------------------- #
# Lorenz
# --------------------------------------------------------------------------- #

@dataclass(frozen = True)
class LorenzConfig:
    '''
    https://en.wikipedia.org/wiki/Lorenz_system
    '''
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8 / 3


def make_lorenz(cfg: LorenzConfig) -> Callable:

    def f(x: Array) -> Array:
        return jnp.stack([
            cfg.sigma * (x[1] - x[0]),
            x[0] * (cfg.rho - x[2]) - x[1],
            x[0] * x[1] - cfg.beta * x[2]
        ])

    return f


def make_reset_lorenz(cfg: LorenzConfig, scale: float = 10.0) -> Callable:

    center = jnp.array([0.0, 0.0, cfg.rho - 1])

    def reset(key):
        return center + jax.random.normal(key, shape = center.shape) * scale

    return reset


# --------------------------------------------------------------------------- #
# Lorenz-96
# --------------------------------------------------------------------------- #

@dataclass(frozen = True)
class Lorenz96Config:
    '''
    https://en.wikipedia.org/wiki/Lorenz_96_model
    '''
    N: int = 10
    F: float = 8.0


def make_lorenz96(cfg: Lorenz96Config) -> Callable:

    def f(x: Array) -> Array:
        return (jnp.roll(x, shift = -1) - jnp.roll(x, shift = 2)) * jnp.roll(x, shift = 1) - x + cfg.F

    return f


def make_reset_lorenz96(cfg: Lorenz96Config, scale: float = 1.0) -> Callable:

    def reset(key):
        return cfg.F + jax.random.normal(key, (cfg.N,)) * scale

    return reset


# --------------------------------------------------------------------------- #
# Rossler
# --------------------------------------------------------------------------- #

@dataclass(frozen = True)
class RosslerConfig:
    '''
    https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
    '''
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7


def make_rossler(cfg: RosslerConfig) -> Callable:

    def f(x: Array) -> Array:
        return jnp.stack([
            -x[1] - x[2],
            x[0] + cfg.a * x[1],
            cfg.b + x[2] * (x[0] - cfg.c)
        ])

    return f


def make_reset_rossler(cfg: RosslerConfig, scale: float = 1.0) -> Callable:
    '''
    Wide resets (scale 3) escape to infinity ~10% of the time; scale 1 measured
    0 escapes in 512 samples.
    '''

    center = jnp.array([0.0, -5.0, 0.0])

    def reset(key):
        return center + jax.random.normal(key, center.shape) * scale

    return reset


# --------------------------------------------------------------------------- #
# Hyperchaotic Rossler
# --------------------------------------------------------------------------- #

@dataclass(frozen = True)
class RosslerHyperchaosConfig:
    '''
    Rossler (1979), An equation for hyperchaos, Phys. Lett. A 71, 155.
    Two positive exponents: roughly (0.11, 0.02, 0, -25) (Wolf et al. 1985).
    '''
    a: float = 0.25
    b: float = 3.0
    c: float = 0.5
    d: float = 0.05


def make_rossler_hyperchaos(cfg: RosslerHyperchaosConfig) -> Callable:

    def f(x: Array) -> Array:
        return jnp.stack([
            -x[1] - x[2],
            x[0] + cfg.a * x[1] + x[3],
            cfg.b + x[0] * x[2],
            -cfg.c * x[2] + cfg.d * x[3]
        ])

    return f


def make_reset_rossler_hyperchaos(cfg: RosslerHyperchaosConfig, scale: float = 0.5) -> Callable:
    '''
    Centered on the initial condition from Wolf et al. (1985). Starts near the other
    commonly quoted point (-10, -6, 0, 10) escape to infinity ~20-35% of the time;
    this box measured 0 escapes in 2048 samples.
    '''

    center = jnp.array([-20.0, 0.0, 0.0, 15.0])

    def reset(key):
        return center + jax.random.normal(key, center.shape) * scale

    return reset


# --------------------------------------------------------------------------- #
# Henon map
# --------------------------------------------------------------------------- #

@dataclass(frozen = True)
class HenonConfig:
    '''
    https://en.wikipedia.org/wiki/H%C3%A9non_map
    '''
    a: float = 1.4
    b: float = 0.3


def make_henon(cfg: HenonConfig) -> Callable:

    def f(x: Array) -> Array:
        return jnp.stack([
            1 - cfg.a * x[0]**2 + x[1],
            cfg.b * x[0]
        ])

    return f


def make_reset_henon(cfg: HenonConfig, half_width: float = 0.5) -> Callable:

    lo = jnp.array([-half_width, -cfg.b * half_width])
    hi = jnp.array([ half_width,  cfg.b * half_width])

    def reset(key):
        return jax.random.uniform(key, (2,), minval = lo, maxval = hi)

    return reset


# --------------------------------------------------------------------------- #
# Ikeda map
# --------------------------------------------------------------------------- #

@dataclass(frozen = True)
class IkedaConfig:
    '''
    https://en.wikipedia.org/wiki/Ikeda_map
    '''
    u: float = 0.9


def make_ikeda(cfg: IkedaConfig) -> Callable:

    def f(x: Array) -> Array:
        t = 0.4 - 6.0 / (1.0 + x[0]**2 + x[1]**2)
        return jnp.stack([
            1.0 + cfg.u * (x[0] * jnp.cos(t) - x[1] * jnp.sin(t)),
            cfg.u * (x[0] * jnp.sin(t) + x[1] * jnp.cos(t))
        ])

    return f


def make_reset_ikeda(cfg: IkedaConfig) -> Callable:
    '''
    The chaotic attractor coexists with a stable fixed point near (2.97, 4.15);
    wide resets get captured by it (a sigma 0.5 Gaussian at the origin loses
    ~10% of trajectories). Uniform starts in this box measured 0 captures in
    65536 samples.
    '''

    lo = jnp.array([0.0, -1.0])
    hi = jnp.array([1.0,  0.0])

    def reset(key):
        return jax.random.uniform(key, (2,), minval = lo, maxval = hi)

    return reset


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass(frozen = True)
class AttractorSpec:
    '''
    Specification of an attractor in the registry; build_attractor(...) turns it into an Attractor.
    '''
    config_cls:     type
    dynamics:       Callable
    reset:          Callable
    dt:             float | None    = None      ## None marks a discrete map
    burn_in:        int             = 0


@dataclass(frozen = True)
class Attractor:
    '''
    The functions used to simulate an attractor, plus the details needed by the estimators.
    '''
    name:       str                 ## name of the attractor
    step:       Callable            ## discrete time update step: X -> X
    f:          Callable            ## continuous time dynamics (the step map itself for discrete maps)
    reset:      Callable            ## key -> a random initial state near the attractor
    dt:         float               ## time discretization if it is a flow (1.0 for maps)
    burn_in:    int                 ## number of steps to run to put the state onto the attractor
    dim:        int                 ## dimensionality of the attractor
    config:     Any                 ## configuration dataclass for the attractor


ATTRACTOR_REGISTRY = {
    'lorenz':               AttractorSpec(LorenzConfig,             make_lorenz,             make_reset_lorenz,             dt = 0.01, burn_in = 1000),
    'lorenz96':             AttractorSpec(Lorenz96Config,           make_lorenz96,           make_reset_lorenz96,           dt = 0.01, burn_in = 500),
    'rossler':              AttractorSpec(RosslerConfig,            make_rossler,            make_reset_rossler,            dt = 0.01, burn_in = 5000),
    'rossler_hyperchaos':   AttractorSpec(RosslerHyperchaosConfig,  make_rossler_hyperchaos, make_reset_rossler_hyperchaos, dt = 0.01, burn_in = 10000),
    'henon':                AttractorSpec(HenonConfig,              make_henon,              make_reset_henon,              dt = None, burn_in = 50),
    'ikeda':                AttractorSpec(IkedaConfig,              make_ikeda,              make_reset_ikeda,              dt = None, burn_in = 100),
}


def build_attractor(
            name: str,
            dt: float | None = None,
            burn_in: int | None = None,
            **cfg_params) -> Attractor:
    '''
    Instantiates an Attractor from its registry name and config parameters.

        build_attractor('lorenz96', N = 50)
        build_attractor('lorenz', dt = 0.005, burn_in = 0, sigma = 16.0, rho = 45.92, beta = 4.0)
    '''

    spec = ATTRACTOR_REGISTRY[name]
    cfg = spec.config_cls(**cfg_params)
    f = spec.dynamics(cfg)
    reset = spec.reset(cfg)
    actual_burn_in = spec.burn_in if burn_in is None else burn_in

    if spec.dt is None:
        if dt is not None:
            raise ValueError(f'{name} is a discrete map so dt is not defined')
        actual_dt = 1.0     ## the dt for maps is one iteration
        step = f
    else:
        actual_dt = spec.dt if dt is None else dt
        step = make_rk4(f, actual_dt)

    ## assumes that attractors operate on vectors
    dim = jax.eval_shape(reset, jax.random.key(0)).shape[0]

    return Attractor(
        name = name,
        step = step,
        f = f,
        reset = reset,
        dt = actual_dt,
        burn_in = actual_burn_in,
        dim = dim,
        config = cfg)
