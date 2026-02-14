# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.interp import cubic_spline_interp, linear_interp
# from val.info import compute_volume


@jax.jit
def compute_volume(fx: Array, fu: Array):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du)

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t, W_t = carry
        fx_t, fu_t = inputs

        ## inverted term
        S_inv = jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t)
        ## feedback gain
        K_t = - S_inv @ fu_t.T @ V_t @ fx_t
        # open loop entropy
        Y_t = Q + fx_t.T @ Y_t @ fx_t
        ## riccati
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t
        ## propagate closed loop entropy
        W_t = Q + (fx_t + fu_t @ K_t).T @ W_t @ (fx_t + fu_t @ K_t)

        carry = (Y_t, V_t, W_t)

        return carry, carry
    
    _, (Y, V, W) = jax.lax.scan(scan_fn, init = (Q, Q, Q), xs = (fx, fu), reverse = True)
    return Y, V, W


class PredictiveSampling:
    def __init__(
            self, 
            dyn: Dynamics, 
            horizon: int, 
            shots: int, 
            knots: int, 
            eps: float, 
            iterations: int, 
            objective: str):
        
        ## actuator control range
        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        ## system parameters
        self.dt = dyn.mjx_model.opt.timestep
        self.horizon = horizon
        self.shots = shots
        self.knots = knots
        self.control_dim = dyn.control_dim
        self.eps = eps
        self.iterations = iterations
        self.objective = objective

        '''
        interpolation
        '''
        # full number of timesteps in the horizon
        self.timesteps = jnp.arange(0, horizon)
        ## break up the horizon into evenly spaced interpolation knots
        self.knot_timesteps = jnp.linspace(0, horizon-1, knots, dtype = int)
        ## batchwise interpolation (knot_timesteps, U_knots, timesteps)
        self.batch_interp = jax.vmap(cubic_spline_interp,  in_axes = (None, 0, None))

        step = make_step(dyn)
        ## linearize a trajectory
        traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
        ## linearize a batch of trajectories
        self.batch_traj_linearize = jax.jit(jax.vmap(traj_linerize))

        ## for unrolling a batch of trajectories with feedback
        unroll = make_unroll(step)
        ## broadcast over actions
        self.batch_unroll = jax.jit(jax.vmap(unroll, in_axes = (None, 0)))

        self.batch_compute_volume = jax.vmap(compute_volume)

        ## initial data
        self.U_knots = jnp.zeros((knots, dyn.control_dim))

    def __call__(self, xt: Array, key):

        U_knots = self.U_knots

        for _ in range(self.iterations):
            key, subkey = jax.random.split(key)

            ## generate noisy knots
            noise = jax.random.normal(subkey, (self.shots, self.knots, self.control_dim))
            U_knots_batch = U_knots[None, :, :] + noise * self.eps
            ## interpolate
            U_batch = self.batch_interp(self.knot_timesteps, U_knots_batch, self.timesteps)

            ## clip controls within range
            U_batch = U_batch.clip(self.low, self.high)

            ## unroll trajectories with feedback
            X_batch = self.batch_unroll(xt, U_batch)

            ## linearize the batch of trajectories
            fx_batch, fu_batch = self.batch_traj_linearize(X_batch[:, :-1, :], U_batch)

            ## compute entropy of each trajectory
            # Y, V = self.batch_compute_volume(fx_batch, fu_batch)
            Y, V, W = self.batch_compute_volume(fx_batch, fu_batch)

            ## compute entropy
            ol_entropy = jnp.linalg.slogdet(Y).logabsdet
            # cl_entropy = jnp.linalg.slogdet(V).logabsdet
            cl_entropy = jnp.linalg.slogdet(W).logabsdet

            ## convert entropy into information
            information = ol_entropy - cl_entropy
            T = jnp.arange(1, horizon + 1)
            rate = information / (jnp.flip(T) * dt)

            # if self.objective == 'rate':
            #     J = rate[:, 0]
            # elif self.objective == 'info':
            #     J = jnp.sum(information, axis = 1)
            # else:
            #     raise NameError

            J = jnp.sum(ol_entropy, axis = 1)

            ## select maximum rate
            idx = jnp.argmax(J)

            U_knots = U_knots_batch[idx]

            # print(J[idx])
            # fig, ax = plt.subplots(2, 1)
            # ax[0].set_ylim(self.low.item(), self.high.item())
            # for i in range(shots):
            #     ax[0].plot(U_batch[i])
            #     ax[1].plot(information[i])
            # plt.show()

        ## select best first action
        ut = U_batch[idx][0]

        ## store optmimzed knots
        self.U_knots = U_knots

        return ut, jnp.mean(J)

if __name__ == '__main__':

    pendulum = 'single'
    
    seed = 1
    key = jax.random.PRNGKey(seed)
    shots = 256

    if pendulum == 'single':
        dt = 0.05
        # works
        horizon = 50
        knots = 20

        # works
        # horizon = 100 ## single
        # knots = 30

        # # sorta works
        # horizon = 200
        # knots = 20

        # # not working
        # horizon = 300
        # knots = 50

        steps = 500 ## single

        dyn = Dynamics('xml/pendulum.xml', dt = dt)

    elif pendulum == 'double':
        dt = 0.01
        ## double pendulum
        horizon = 600
        knots = 50
        steps = 3000

        dyn = Dynamics('xml/double_pendulum.xml', dt = dt)

    step = make_step(dyn)

    eps = 0.1
    iterations = 1
    alpha = 1.0
    objective = 'info'

    mpc = PredictiveSampling(dyn, horizon, shots, knots, eps, iterations, objective)

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

    name = f'PS-{objective}-{pendulum}_pendulum-h={horizon}-shots={shots}-knots={knots}-eps={eps}-iterations={iterations}-alpha={alpha}-dt={dt}'
    
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