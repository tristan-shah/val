from typing import Callable

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.utils import estimate_lyapunov_hist

def make_batch_compute_max_lyapunov(step: Callable, dt: float):

    unroll = make_unroll(step)
    batch_unroll = jax.vmap(unroll, in_axes = (None, 0))
    traj_linearize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))
    batch_traj_linearize = jax.vmap(traj_linearize)
    batch_estimate_lyapunov_hist = jax.vmap(estimate_lyapunov_hist, in_axes = (0, None))

    def batch_compute_max_lyapunov(xt: Array, U: Array):

        X = batch_unroll(xt, U)
        fx, _ = batch_traj_linearize(X[:, :-1, :], U)

        ## actual LCE
        ## 3D array (shots x horizon x state_dim)
        lce = batch_estimate_lyapunov_hist(fx, dt)


        '''
        For SUPLE
        '''
        # ## convert to sum of positive LE
        # ## (shots x horizon)
        # suple = jnp.sum(jnp.clip(lce, min = 0.0), axis = 2)

        # ## sum over trajectory
        # ## (shots,)
        # sum_max_lce = jnp.mean(suple, axis = 1)
        # return sum_max_lce, X

        '''
        For volume
        '''
        # horizon = U.shape[1]
        # t = jnp.arange(1, horizon + 1).reshape(1, -1, 1)
        # vol = jnp.log(jnp.sum(jnp.exp(lce * t * dt), axis = (1, 2)))
        # return vol, X

        '''
        Final largest LCE
        '''
        last_lce = lce[:, -1, :]
        max_last_lce = last_lce.max(axis = -1)
        return max_last_lce, X

    return jax.jit(batch_compute_max_lyapunov)

class MPC:
    def __init__(self, dyn: Dynamics, shots: int, horizon: int, cost: Callable):

        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]
        self.U_bar = jnp.zeros((horizon, dyn.control_dim))
        self.shots = shots
        self.horizon = horizon

        self.batch_cost = jax.vmap(cost, in_axes = (None, 0))

    def __call__(self, xt: Array, key):
        
        U = self.U_bar[None, :, :] + jax.random.normal(key, shape = (self.shots, self.horizon, dyn.control_dim)) * eps
        U = U.clip(self.low, self.high)

        # lce, _ = self.batch_compute_max_lyapunov(xt, U)
        # idx = jnp.argmax(lce)
        J = self.batch_cost(xt, U)
        idx = jnp.argmax(J)

        self.U_bar = U[idx]
        ut = self.U_bar[0]
        self.U_bar = jnp.roll(self.U_bar, shift = -1, axis = 0)
        return ut, J[idx]
    
@jax.jit
def compute_feedback_gains(fx: Array, fu: Array):
    dx, du = fx.shape[-1], fu.shape[-1]
    cxx = jnp.eye(dx)
    cuu = jnp.eye(du) * 1.0

    def scan_fn(Vxx: Array, inputs: tuple):

        fx_t, fu_t = inputs
        S_inv = jnp.linalg.inv(cuu + fu_t.T @ Vxx @ fu_t)
        K_t = - S_inv @ fu_t.T @ Vxx @ fx_t

        D = fx_t + fu_t @ K_t
        Vxx = cxx + K_t.T @ cuu @ K_t + D.T @ Vxx @ D
        return Vxx, K_t
    
    _, K = jax.lax.scan(scan_fn, init = cxx, xs = (fx, fu), reverse = True)
    return K

'''
I need to compare the singular values of open and closed loop derivatives. We only care about minimizing directions of expansion in open loop.
We can align the closed loop singular values with the open loop singular values and see how much the controller is surpressing chaos.
'''

@jax.jit
def compute_vol(fx: Array):
    dx = fx.shape[-1]
    cxx = jnp.eye(dx)

    def scan_fn(Vxx_t: Array, fx_t: Array):

        Vxx_t = cxx + fx_t.T @ Vxx_t @ fx_t
        return Vxx_t, Vxx_t
    
    Vxx, _ = jax.lax.scan(scan_fn, init = cxx, xs = fx, reverse = True)
    return Vxx

def make_compute_gap(step: Callable, dt: float):

    unroll = make_unroll(step)
    traj_linearize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))

    def compute_gap(xt: Array, U: Array):

        X = unroll(xt, U)
        fx, fu = traj_linearize(X[:-1], U)

        '''
        Try to sum everything
        '''
        # log_svd = jnp.log(jnp.linalg.svdvals(fx))
        # t = jnp.exp(-jnp.arange(U.shape[0], 0, -1).reshape(-1, 1))
        # t = jnp.arange(1, U.shape[0] + 1).reshape(-1, 1)
        # mask = log_svd > 0
        # return jnp.sum(t * log_svd * mask)

        '''
        suple variants
        '''
        # lce_uncontrolled = estimate_lyapunov_hist(fx, dt)
        # K = compute_feedback_gains(fx, fu)
        # D = fx + einsum(fu, K, 't x1 u, t u x2 -> t x1 x2')
        # # lce_controlled = estimate_lyapunov_hist(D, dt)
        # mask = lce_uncontrolled > 0.0
        # # gap = jnp.sum(lce_uncontrolled * mask - lce_controlled * mask)
        # suple = jnp.sum(lce_uncontrolled * mask)
        # # suple = jnp.sum(lce_uncontrolled.max(axis = -1))
        # return suple
    
        '''
        Volume
        '''
        # vol = 0.5 * jnp.log(jnp.linalg.eigvalsh(compute_vol(fx)))
        # return 0.5 * jnp.log(jnp.trace(compute_vol(fx)))
        return jnp.log(jnp.linalg.det(compute_vol(fx)))

    return jax.jit(compute_gap)

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    ## environment parameters
    dt = 0.05
    horizon = 100
    eps = 0.1
    shots = 128
    steps = 800

    dyn = Dynamics('xml/pendulum.xml', dt = dt, integrator = 'euler')
    # dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
    step = make_step(dyn)
    compute_gap = make_compute_gap(step, dt)

    mpc = MPC(dyn, shots, horizon, cost = compute_gap)
   
    '''
    Testing random shit
    '''
    # U_bar = jnp.zeros((horizon, dyn.control_dim))
    # U = U_bar[None, :, :] + jax.random.normal(key, shape = (shots, horizon, dyn.control_dim)) * eps
    # U = U.clip(mpc.low, mpc.high)

    # X = batch_unroll(xt, U)
    # fx, fu = batch_traj_linearize(X[:, :-1, :], U)
    # lce = batch_estimate_lyapunov_hist(fx, dt)

    # t = jnp.arange(1, horizon + 1).reshape(1, -1, 1)
    # vol = jnp.sum(jnp.exp(lce * t * dt), axis = (1, 2))
    # print(vol.shape)

    # fig, ax = plt.subplots(1, 1)
    # for i in range(shots):
    #     ax.plot(lce[i, :, 0])
    # plt.show()

    # print(compute_gap(xt, U_bar))







    # '''
    # Comparing open and closed loop singular values
    # '''
    # U_ol, sigma_ol, Vh_ol = jnp.linalg.svd(fx)
    # D_tilde = einsum(U_ol, D, Vh_ol, 'b x x1, b x y, b x2 y -> b x1 x2')
    
    # sigma_cl = jnp.linalg.svdvals(D_tilde)

    # log_sigma_cl = jnp.log(sigma_cl)
    # log_sigma_ol = jnp.log(sigma_ol)
    
    # fig, ax = plt.subplots(dyn.state_dim, 1)
    
    # for i in range(dyn.state_dim):
    #     ax[i].plot(log_sigma_ol[:, i])
    #     ax[i].plot(log_sigma_cl[:, i])
    # plt.show()





    

    # t = jnp.arange(1, horizon + 1).reshape(-1, 1)
    # vol = jnp.exp(dt * t * lce)
    # print(vol.sum())

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(jnp.log(vol[:, 0]))
    # ax.plot(jnp.log(vol[:, 1]))
    # ax.plot(jnp.log(vol.sum(axis = -1)))
    # plt.show()

    # dyn.render(X, 'test.mp4', skip = 1)
    # dyn.render(X, path = 'double.mp4', distance = 5.0, skip = 1, lookat = jnp.array([0, 0, 2.2]))













    # '''
    # Warmstart run open loop sampling
    # '''
    # ## initial control sequence
    # U_bar = jnp.zeros((horizon, dyn.control_dim))

    # hist = []

    # for i in range(100):
    #     U = U_bar[None, :, :] + jax.random.normal(key, shape = (shots, horizon, dyn.control_dim)) * eps
    #     U = jnp.concatenate([U_bar[None, :, :], U])

    #     U = U.clip(mpc.low, mpc.high)
    #     lce, _ = mpc.batch_compute_max_lyapunov(xt, U)
    #     idx = jnp.argmax(lce)
    #     U_bar = U[idx]
    #     print(i, lce[idx])
    #     hist.append(lce[idx])

    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(hist)
    # ax[1].plot(U_bar)
    # plt.show()

    '''
    Run closed loop PS
    '''
    xt = jnp.zeros(dyn.state_dim)
    xt = xt.at[0].set(0.2)

    # linearize = jax.jacfwd(step, argnums = (0, 1))
    # traj_linearize = jax.vmap(linearize)
    # U = jnp.zeros((horizon, dyn.control_dim))
    # unroll = make_unroll(step)
    # X = unroll(xt, U)
    # fx, fu = traj_linearize(X[:-1], U)
    # # vol = 0.5 * jnp.log(jnp.linalg.eigvalsh(compute_vol(fx)))
    # Vxx = compute_vol(fx)



    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    hist = []

    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J = mpc(xt, subkey)
        xt = step(xt, ut)
        print(t, xt, ut, J)
        X = X.at[t+1].set(xt)
        hist.append(J)

    dyn.render(X, 'test.mp4', skip = 1)
    # dyn.render(X, path = 'double.mp4', distance = 5.0, skip = 1, lookat = jnp.array([0, 0, 2.2]))

    fig, ax = plt.subplots(1, 1)
    ax.plot(hist)
    plt.show()