# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.interp import cubic_spline_interp

def make_compute_k(horizon: int, dim: int, shots: int, knots: int):

    # full number of timesteps in the horizon
    timesteps = jnp.arange(0, horizon)
    ## break up the horizon into evenly spaced interpolation knots
    ## indices of the knots
    knots_idx = jnp.linspace(0, horizon-1, knots, dtype = int)

    ## batchwise interpolation
    batch_interp = jax.vmap(cubic_spline_interp,  in_axes = (None, 0, None))

    def compute_k(key):
        noise = jax.random.normal(key, (shots, knots, dim))
        k = batch_interp(knots_idx, noise, timesteps)
        return k
    
    return jax.jit(compute_k)

@jax.jit
def compute_volume(fx: Array, fu: Array, alpha: float):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du) * alpha
    gamma = 1.0

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t = carry
        fx_t, fu_t = inputs

        S_inv = jnp.linalg.inv(R + gamma * fu_t.T @ V_t @ fu_t)
        
        K_t = - gamma * S_inv @ fu_t.T @ V_t @ fx_t
        Y_t = Q + gamma * fx_t.T @ Y_t @ fx_t
        V_t = Q + gamma * fx_t.T @ V_t @ fx_t - (gamma**2) * fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t

        carry = (Y_t, V_t)

        return carry, (Y_t, V_t, K_t)
    
    _, (Y, V, K) = jax.lax.scan(scan_fn, init = (Q, Q), xs = (fx, fu), reverse = True)
    return Y, V, K

def make_unroll_feedback(step: callable, low: Array, high: Array):

    def unroll_feedback(x0: Array, X_bar: Array, U_bar: Array, K: Array):

        def body_fn(x: Array, inputs: tuple[Array, Array]):

            x_bar, u_bar, K_t = inputs
            ## apply state feedback
            u_feedback = u_bar + K_t @ (x - x_bar)
            u_feedback = u_feedback.clip(low, high)

            ## step dynamics
            x_next = step(x, u_feedback)

            return x_next, (x_next, u_feedback)
        
        _, (X, U_feedback) = jax.lax.scan(body_fn, x0, xs = (X_bar[:-1], U_bar, K))
        X = jnp.concatenate([x0[None, :], X], axis = 0)

        return X, U_feedback
    
    return jax.jit(unroll_feedback)

class FeedbackMPC:
    def __init__(self, dyn: Dynamics, horizon: int, shots: int, knots: int, eps: float, iterations: int, alpha: float):

        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        self.dt = dyn.mjx_model.opt.timestep
        self.horizon = horizon
        self.shots = shots
        self.control_dim = dyn.control_dim
        self.eps = eps
        self.iterations = iterations
        self.alpha = alpha

        self.compute_k = make_compute_k(horizon, dyn.control_dim, shots, knots)

        step = make_step(dyn)
        ## linearize a trajectory
        traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
        ## linearize a batch of trajectories
        self.batch_traj_linearize = jax.jit(jax.vmap(traj_linerize))

        ## for unrolling a batch of trajectories with feedback
        unroll_feedback = make_unroll_feedback(step, self.low, self.high)
        ## broadcast over actions
        self.batch_unroll_feedback = jax.jit(jax.vmap(unroll_feedback, in_axes = (None, None, 0, None)))

        self.batch_compute_volume = jax.vmap(compute_volume, in_axes = (0, 0, None))

        ## initial data
        self.X = jnp.zeros((horizon + 1, dyn.state_dim))
        self.U = jnp.zeros((horizon, dyn.control_dim))
        self.K = jnp.zeros((horizon, dyn.control_dim, dyn.state_dim))

    def __call__(self, xt: Array, key):

        X = self.X
        U = self.U
        K = self.K

        for _ in range(self.iterations):
            key, subkey = jax.random.split(key)
            
            ## sample perturbations
            k_batch = self.compute_k(subkey) * self.eps

            ## apply perturbations to nominal action sequence
            U_batch = U[None, :, :] + k_batch
            U_batch = U_batch.clip(self.low, self.high)

            ## unroll trajectories with feedback
            X_batch, U_batch = self.batch_unroll_feedback(xt, X, U_batch, K)

            ## linearize the batch of trajectories
            fx_batch, fu_batch = self.batch_traj_linearize(X_batch[:, :-1, :], U_batch)

            ## compute entropy of each trajectory
            Y, V, K_batch = self.batch_compute_volume(fx_batch, fu_batch, self.alpha)
            K_batch = K_batch ## turn off feedback

            ## compute entropy
            ol_entropy = jnp.linalg.slogdet(Y).logabsdet
            cl_entropy = jnp.linalg.slogdet(V).logabsdet

            ## convert entropy into information
            information = ol_entropy - cl_entropy
            T = jnp.arange(1, horizon + 1)
            rate = information / (jnp.flip(T) * dt)
            # J = rate[:, 0]
            J = jnp.sum(information, axis = 1)

            # '''
            # MPPI implementation
            # '''
            # temp = 10.0
            # # 1. Compute weights
            # J_min = J.min()
            # weights = jnp.exp(-(J - J_min) / temp)
            # weights /= jnp.sum(weights)

            # # 2. Update controls as weighted average
            # U = jnp.sum(weights[:, None, None] * U_batch, axis = 0)



            ## select maximum rate
            idx = jnp.argmax(J)

            X = X_batch[idx]
            U = U_batch[idx]
            K = K_batch[idx]

            # print(J[idx])

            # fig, ax = plt.subplots(2, 1)
            # ax[0].set_ylim(self.low.item(), self.high.item())
            # for i in range(shots):
            #     ax[0].plot(U_batch[i])
            #     ax[1].plot(information[i])
            # plt.show()

        ## select first action

        ut = U[0]
        ## shift over the plan by one action
        X = jnp.roll(X, shift = -1, axis = 0)
        U = jnp.roll(U, shift = -1, axis = 0)
        K = jnp.roll(K, shift = -1, axis = 0)

        X = X.at[-1].set(X[-2])
        U = U.at[-1].set(U[-2])
        K = K.at[-1].set(K[-2])

        self.X = X
        self.U = U
        self.K = K

        return ut, J[idx] #jnp.mean(first_rate)

if __name__ == '__main__':

    pendulum = 'double'
    
    seed = 0
    key = jax.random.PRNGKey(seed)
    shots = 512

    if pendulum == 'single':
        dt = 0.05
        # works
        horizon = 50
        knots = 10

        # # works
        # horizon = 100 ## single
        # knots = 10

        # # sorta works
        # horizon = 200
        # knots = 20

        # # not working
        # horizon = 300
        # knots = 30
        steps = 500 ## single

        dyn = Dynamics('xml/pendulum.xml', dt = dt)

    elif pendulum == 'double':
        dt = 0.01
        ## double pendulum
        horizon = 600
        knots = 60
        steps = 3000

        dyn = Dynamics('xml/double_pendulum.xml', dt = dt)


    eps = 0.2
    iterations = 5
    alpha = 10.0

    step = make_step(dyn)

    mpc = FeedbackMPC(dyn, horizon, shots, knots, eps, iterations, alpha)

    xt = jnp.zeros(dyn.state_dim)

    # mpc(xt, key)

    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = jnp.zeros(steps)

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J = mpc(xt, subkey)
        xt = step(xt, ut)
        print(t, xt, ut, J)

        X = X.at[t+1].set(xt)
        hist = hist.at[t].set(J)

    # name = f'single_pendulum-h={horizon}-shots={shots}-knots={knots}-eps={eps}-iterations={iterations}-alpha={alpha}-dt={dt}'
    name = f'{pendulum}_pendulum-h={horizon}-shots={shots}-knots={knots}-eps={eps}-iterations={iterations}-alpha={alpha}-dt={dt}'

    
    T = jnp.arange(0, steps)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(name + '.png', dpi = 300)
    plt.show()

    if pendulum == 'single':
        dyn.render(X, path = name + '.mp4', skip = 1)
    elif pendulum == 'double':
        dyn.render(X, path = name + '.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))