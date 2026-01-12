import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
# from val.info import make_compute_entropy


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

def make_compute_entropy(step: callable):
    
    traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    unroll = make_unroll(step)

    def compute_entropy(x0: Array, U: Array):

        X = unroll(x0, U)
        fx, fu = traj_linerize(X[:-1], U)
        Y, V, W = compute_volume(fx, fu)

        ol_entropy = jnp.linalg.slogdet(Y).logabsdet
        cl_entropy = jnp.linalg.slogdet(V).logabsdet
        # cl_entropy = jnp.linalg.slogdet(W).logabsdet

        return ol_entropy, cl_entropy
    
    return jax.jit(compute_entropy)



class MPC:
    def __init__(self, dyn: Dynamics, horizon: int, shots: int, eps: float, iterations: int, window: int = 1):

        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        self.dt = dyn.mjx_model.opt.timestep
        self.horizon = horizon
        self.shots = shots
        self.control_dim = dyn.control_dim
        self.eps = eps
        self.iterations = iterations
        self.window = window

        self.compute_entropy = jax.jit(jax.vmap(make_compute_entropy(make_step(dyn)), in_axes = (None, 0)))
        self.U = jnp.zeros((horizon, dyn.control_dim))

    def __call__(self, xt: Array, key):

        U = self.U

        for _ in range(self.iterations):

            key, subkey = jax.random.split(key)

            U_batch = U[None, :, :] + (jax.random.normal(subkey, (self.shots, self.horizon, self.control_dim)) * self.eps)
            U_batch = U_batch.clip(self.low[None, None, :], self.high[None, None, :])
            
            ## compute the gap along each trajectory
            ol_entropy, cl_entropy = self.compute_entropy(xt, U_batch)
            information = ol_entropy - cl_entropy

            ## normalize gap by time
            T = jnp.arange(1, self.horizon + 1)
            rate = information / (jnp.flip(T) * self.dt)

            ## average over time (usually just select the first one)
            J = rate[:, 0]
            # J = jnp.sum(information, axis = -1)

            # fig, ax = plt.subplots(2, 1)
            # ax[0].set_ylim(self.low.item(), self.high.item())
            # for i in range(shots):
            #     ax[0].plot(U_batch[i])
            #     ax[1].plot(information[i])
            # plt.show()

            ## hard selection
            idx = jnp.argmax(J)
            U = U_batch[idx]
            # print(J[idx])


        ut = U[0]
        ## shift over the plan by one action
        U = jnp.roll(U, shift = -1, axis = 0)
        U = U.at[-1].set(U[-2])
        self.U = U

        return ut, jnp.mean(J)

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    dt = 0.01 ## double
    # dt = 0.05 ## single
    horizon = 600
    # horizon = 300 ## single
    shots = 2048
    eps = 0.1
    iterations = 5
    # steps = 500 ## single
    steps = 3000 ## double
    window = 1

    # dyn = Dynamics('xml/pendulum.xml', dt = dt)
    dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
    step = make_step(dyn)

    theta = 0.0
    x0 = jnp.zeros(dyn.state_dim)
    xt = x0.copy()

    mpc = MPC(dyn, horizon, shots, eps, iterations, window)

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

    # name = f'single_pendulum-iterations={iterations}-dt={dt}-h={horizon}'
    name = f'double_pendulum-shots-{shots}-eps={eps}-iterations={iterations}-dt={dt}-h={horizon}'
    
    jnp.save(name + '.npy', hist)

    T = jnp.arange(0, steps)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats / s')
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(name + '.png', dpi = 300)
    plt.show()

    # dyn.render(X, path = name + '.mp4', skip = 1)
    dyn.render(X, path = name + '.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))


