import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.interp import linear_interp, cubic_spline_interp
from val.info import make_compute_entropy

class MPC:
    def __init__(self, dyn: Dynamics, horizon: int, knots: int, shots: int, eps: float, iterations: int):

        ## range of the action space
        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        ## time discretization
        self.dt = dyn.mjx_model.opt.timestep

        ## planning horizon
        self.horizon = horizon
        ## discretization of planning horizon
        self.knots = knots
        ## number of trajectories to evaluate
        self.shots = shots
        
        ## dimention of the control space
        self.control_dim = dyn.control_dim
        
        ## exploration noise
        self.eps = eps

        ## number of predictive sampling iterations per step
        self.iterations = iterations

        ## evaluate entropy of each trajectory in the batch
        self.batch_compute_entropy = jax.jit(jax.vmap(make_compute_entropy(make_step(dyn)), in_axes = (None, 0)))

        ## initial plan (zero)
        self.U = jnp.zeros((horizon, dyn.control_dim))

        '''
        interpolation
        '''
        # full number of timesteps in the horizon
        self.timesteps = jnp.arange(0, horizon)
        ## break up the horizon into evenly spaced interpolation knots
        ## indices of the knots
        self.knots_idx = jnp.linspace(0, horizon-1, knots, dtype = int)

    def __call__(self, xt: Array, key):
        U = self.U

        for _ in range(self.iterations):
            ## extract data at current knot values
            U_knots = U[self.knots_idx, :]
            ## generate exploration noise
            noise = jax.random.normal(key, (self.shots, self.knots, self.control_dim))
            ## apply exploration noise
            U_knots_batch = (U_knots[None, :, :] + self.eps * noise).clip(self.low, self.high)

            U_batch = jax.vmap(cubic_spline_interp, in_axes = (None, 0, None))(self.knots_idx, U_knots_batch, self.timesteps).clip(self.low, self.high)
            # U_batch = jax.vmap(linear_interp, in_axes = (None, 0, None))(self.knots_idx, U_knots_batch, self.timesteps).clip(self.low, self.high)

            ## compute the gap along each trajectory
            ol_entropy, cl_entropy = self.batch_compute_entropy(xt, U_batch)
            gap = ol_entropy - cl_entropy

            ## normalize by time to get information rate
            rate = gap# / (jnp.flip(self.timesteps + 1) * dt)
            # first_rate = rate[:, 0]
            first_rate = jnp.sum(rate, axis = 1)
            ## find trajectory with maximum rate
            idx = jnp.argmax(first_rate)

            ## select the best control sequence
            U = U_batch[idx]

            # visualize planning
            fig, ax = plt.subplots(2, 1)
            ax[0].set_ylim(self.low.item(), self.high.item())
            for i in range(self.shots):
                ax[0].plot(U_batch[i])
                ax[1].plot(rate[i])
            plt.show()

        ut = U[0]
        ## shift over the plan by one action
        U = jnp.roll(U, shift = -1, axis = 0)
        U = U.at[-1].set(U[-2])
        self.U = U
        return ut, jnp.mean(first_rate)

if __name__ == '__main__':
    
    seed = 0
    key = jax.random.PRNGKey(seed)
    dt = 0.05 ## single
    horizon = 300 ## single
    knots = 30 #horizon // 2
    shots = 128
    eps = 0.1
    iterations = 20
    steps = 600 ## single

    dyn = Dynamics('xml/pendulum.xml', dt = dt)
    mpc = MPC(dyn, horizon, knots, shots, eps, iterations)
    step = make_step(dyn)

    xt = jnp.zeros(dyn.state_dim)
    # X = jnp.zeros((steps + 1, dyn.state_dim))
    # X = X.at[0].set(xt)

    mpc(xt, key)

    # hist = jnp.zeros(steps)

    # for t in range(steps):
    #     key, subkey = jax.random.split(key)
    #     ut, J = mpc(xt, subkey)
    #     xt = step(xt, ut)
    #     print(t, xt, ut, J)

    #     X = X.at[t+1].set(xt)
    #     hist = hist.at[t].set(J)

    # name = f'single_pendulum-iterations={iterations}-dt={dt}-h={horizon}'

    # T = jnp.arange(0, steps)

    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('nats / s')
    # ax.plot(T * dt, hist)
    # fig.tight_layout()
    # fig.savefig(name + '.png', dpi = 300)
    # plt.show()

    # dyn.render(X, path = name + '.mp4', skip = 1)