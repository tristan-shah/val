'''
Controllable Information Production (CIP) over a finite planning horizon.

Given a rollout X of a control sequence U, the open-loop and closed-loop information production
rates (Section 3.2) are the time-normalized log-determinants of the recursions in Equation (10),
propagated backwards along the trajectory in their numerically stable inverse form.

Following Definition 3.5, CIP is the open-loop rate. The closed-loop rate and the difference
between the two are also returned: the difference was the working definition of CIP early in
the project and is kept so that every quantity stays available for analysis.
'''

import jax
from jax import Array
from jax import numpy as jnp

from val import Dynamics, make_step, make_unroll


def make_compute_entropy(step: callable):
    '''
    Returns a jitted function (X, U) -> (logdet_Y, logdet_W) that accumulates, backwards in
    time, the open-loop (Y) and closed-loop (W) log-determinants of Equation (10). Jacobians
    are formed inside the scan so only one pair is ever materialized.
    '''

    linearize = jax.jacfwd(step, argnums = (0, 1))

    def compute_entropy(X: Array, U: Array):

        assert X.shape[0] == U.shape[0]

        dx = X.shape[-1]
        du = U.shape[-1]

        Q = jnp.eye(dx)
        R = jnp.eye(du)

        def scan_fn(carry: tuple, inputs: tuple[Array, Array]):
            ## logdet_Y_t, logdet_W_t: accumulated log det of Y_t and W_t
            ## Y_inv_t, W_inv_t:       inverses of Y_t and W_t
            ## V_t:                    solution of the Riccati equation
            logdet_Y_t, logdet_W_t, Y_inv_t, W_inv_t, V_t = carry
            x_t, u_t = inputs
            fx_t, fu_t = linearize(x_t, u_t)

            S_inv = jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t)
            ## optimal feedback gain and closed-loop dynamics
            K_t = - S_inv @ fu_t.T @ V_t @ fx_t
            D_t = fx_t + fu_t @ K_t

            ## matrices to invert
            M_ol = Y_inv_t + fx_t @ fx_t.T
            M_cl = W_inv_t + D_t @ D_t.T

            ## accumulate log determinants
            logdet_Y_t = logdet_Y_t + jnp.linalg.slogdet(M_ol).logabsdet
            logdet_W_t = logdet_W_t + jnp.linalg.slogdet(M_cl).logabsdet

            ## propagate inverses
            Y_inv_t = Q - fx_t.T @ jnp.linalg.inv(M_ol) @ fx_t
            W_inv_t = Q - D_t.T @ jnp.linalg.inv(M_cl) @ D_t

            ## propagate Riccati
            V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t

            carry = (logdet_Y_t, logdet_W_t, Y_inv_t, W_inv_t, V_t)
            return carry, (logdet_Y_t, logdet_W_t)

        init = (jnp.array(0.0), jnp.array(0.0), Q, Q, Q)
        _, (logdet_Y, logdet_W) = jax.lax.scan(scan_fn, init = init, xs = (X, U), reverse = True)
        return logdet_Y, logdet_W

    return jax.jit(compute_entropy)


OBJECTIVES = ['cip', 'closed_loop', 'difference']

## results folder for runs of each objective; the paper's runs are the 'cip' objective
RESULTS_FOLDER = {'cip': 'CIP', 'closed_loop': 'CIP_CLOSED_LOOP', 'difference': 'CIP_DIFFERENCE'}


def make_compute_cip_from_step(step: callable, dt: float, objective: str = 'cip'):
    '''
    Builds the planning objective from a jax `step` callable and the time step `dt`.
    Returns a jitted function (xt, U) -> (J, info) with info = {'cip', 'closed_loop', 'difference'}
    in nats / s and J the quantity to maximize:

        'cip'           the open-loop information production rate (the paper's objective)
        'closed_loop'   the negated closed-loop rate, i.e. minimize closed-loop production
        'difference'    open-loop minus closed-loop rate
    '''

    assert objective in OBJECTIVES

    unroll = make_unroll(step)
    compute_entropy = make_compute_entropy(step)

    def compute_cip(xt: Array, U: Array):
        X = unroll(xt, U)

        logdet_Y, logdet_W = compute_entropy(X[:-1], U)

        horizon = U.shape[0]
        T = jnp.arange(1, horizon + 1)

        ## normalize by the remaining time at each step and keep the full-horizon value
        cip = (logdet_Y / (2 * jnp.flip(T) * dt))[0]            ## open-loop rate
        closed_loop = (logdet_W / (2 * jnp.flip(T) * dt))[0]    ## closed-loop rate

        info = {'cip': cip, 'closed_loop': closed_loop, 'difference': cip - closed_loop}

        if objective == 'cip':
            return cip, info
        elif objective == 'closed_loop':
            return -closed_loop, info
        elif objective == 'difference':
            return cip - closed_loop, info

    return jax.jit(compute_cip)


def make_compute_cip(dyn: Dynamics, objective: str = 'cip'):
    '''
    Same as `make_compute_cip_from_step` for a MuJoCo `Dynamics`.
    '''
    return make_compute_cip_from_step(make_step(dyn), dyn.mjx_model.opt.timestep, objective)
