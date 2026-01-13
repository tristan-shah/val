import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['MUJOCO_GL'] = 'egl'

import jax
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

def ar1_noise(key, shots: int, horizon: int, control_dim: int, rho: float = 0.9):
    '''
    Returns noise of shape (shots, horizon, control_dim)
    with AR(1) time correlation.
    '''
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (horizon, shots, control_dim))

    alpha = jnp.sqrt(1.0 - rho ** 2)

    def step(prev, curr):
        out = rho * prev + alpha * curr
        return out, out

    init = eps[0]                      # (shots, control_dim)
    _, ys = jax.lax.scan(step, init, eps[1:])

    noise = jnp.concatenate(
        [init[None, ...], ys], axis=0
    )                                   # (horizon, shots, control_dim)

    return jnp.transpose(noise, (1, 0, 2))

@jax.jit
def compute_volume(fx: Array, fu: Array, alpha: float):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)
    R = jnp.eye(du) * alpha

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t, W_t = carry
        fx_t, fu_t = inputs

        S_inv = jnp.linalg.inv(R + fu_t.T @ V_t @ fu_t)
        ## feedback gain
        K_t = - S_inv @ fu_t.T @ V_t @ fx_t
        ## open loop entropy
        Y_t = Q + fx_t.T @ Y_t @ fx_t
        ## riccati equation
        V_t = Q + fx_t.T @ V_t @ fx_t - fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t
        ## closed loop entropy
        W_t = Q + (fx_t + fu_t @ K_t).T @ W_t @ (fx_t + fu_t @ K_t)

        carry = (Y_t, V_t, W_t)

        return carry, (Y_t, V_t, W_t, K_t)
    
    _, (Y, V, W, K) = jax.lax.scan(scan_fn, init = (Q, Q, Q), xs = (fx, fu), reverse = True)
    return Y, V, W, K

OBJECTIVES = ['rate', 'mean_rate', 'mean_info', 'mean_neg_cl_ent']

def make_compute_rate(dyn: Dynamics, objective: str, alpha: float):

    assert objective in OBJECTIVES

    ## helper functions
    step = make_step(dyn)
    traj_linerize = jax.jit(jax.vmap(jax.jacfwd(step, argnums = (0, 1))))
    batch_traj_linearize = jax.jit(jax.vmap(traj_linerize))
    unroll = make_unroll(step)
    batch_unroll = jax.jit(jax.vmap(unroll, in_axes = (None, 0)))
    batch_compute_volume = jax.jit(jax.vmap(compute_volume, in_axes = (0, 0, None)))

    dt = dyn.mjx_model.opt.timestep

    def compute_rate(xt: Array, U_batch: Array):

        ## unroll trajectories
        X_batch = batch_unroll(xt, U_batch)
        ## linearize the batch of trajectories
        fx_batch, fu_batch = batch_traj_linearize(X_batch[:, :-1, :], U_batch)
        # ## compute entropy of each trajectory (manually setting alpha = 1.0)
        Y, V, W, _ = batch_compute_volume(fx_batch, fu_batch, alpha)
        ## compute entropy
        ol_entropy = jnp.linalg.slogdet(Y).logabsdet
        # cl_entropy = jnp.linalg.slogdet(V).logabsdet
        cl_entropy = jnp.linalg.slogdet(W).logabsdet
        ## convert entropy into information
        information = ol_entropy - cl_entropy

        horizon = U_batch.shape[1]
        T = jnp.arange(1, horizon + 1)
        rate = information / (jnp.flip(T) * dt)

        if objective == 'rate':
            return rate[:, 0]
        elif objective == 'mean_rate':
            return jnp.mean(rate, axis = 1)
        elif objective == 'mean_info':
            return jnp.mean(information, axis = 1)
        elif objective == 'mean_neg_cl_ent':
            return jnp.mean(-cl_entropy, axis = 1)

    return jax.jit(compute_rate)

class CEM:
    def __init__(
            self, 
            dyn: Dynamics, 
            objective: callable,
            shots: int, 
            horizon: int, 
            iterations: int, 
            elite_frac: float,
            smoothing: float):
        
        assert 0.0 < elite_frac <= 1.0

        ## actuator ranges
        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        ## objective function
        self.objective = objective

        ## planning parameters
        self.shots = shots
        self.horizon = horizon
        self.control_dim = dyn.control_dim
        self.iterations = iterations
        self.n_elite = max(1, int(elite_frac * shots))
        self.smoothing = smoothing

        ## set a minimum exploration amount
        self.min_std = 0.05 * (self.high - self.low) ## default

        ## initial mean and std
        self.mean = jnp.zeros((self.horizon, self.control_dim))
        self.std = jnp.ones((self.horizon, self.control_dim))

    def __call__(self, xt: Array, key):

        hist = []
        for i in range(self.iterations):

            ## generate a batch of random control signals
            key, subkey = jax.random.split(key)
            # noise = jax.random.normal(subkey, (self.shots, self.horizon, self.control_dim))
            noise = ar1_noise(subkey, self.shots, self.horizon, self.control_dim)

            U_batch = self.mean[None, :, :] + self.std[None, :, :] * noise
            U_batch = U_batch.clip(self.low[None, None, :], self.high[None, None, :])

            ## evaluate control signals in parallel
            J = self.objective(xt, U_batch)

            ## select top control sequences
            elite_idx = jnp.argsort(J, descending = True)[:self.n_elite]
            U_elite = U_batch[elite_idx]

            ## fit gaussian
            new_mean = jnp.mean(U_elite, axis = 0)
            new_std = jnp.std(U_elite, axis = 0)

            ## smooth update
            self.mean = (1 - self.smoothing) * new_mean + self.smoothing * self.mean
            self.std  = (1 - self.smoothing) * new_std  + self.smoothing * self.std
            ## maintain minimum exploration
            self.std = self.std.clip(min = self.min_std)

        #     max_J = jnp.max(J)
        #     print(max_J)
        #     hist.append(max_J)

        # fig, ax = plt.subplots(1, 3, figsize = (10, 8))

        # fig.suptitle(f'Cross Entropy Method. Objective = {objective_type}')
        # ax[0].set_xlabel('Iteration')

        # if objective_type == 'rate' or objective_type == 'mean_rate':
        #     ax[0].set_ylabel('nats / s')
        # elif objective_type == 'mean_info':
        #     ax[0].set_ylabel('nats')

        # ax[1].set_xlabel('Horizon')
        # ax[1].set_ylabel('Control Signal')

        # ax[2].set_xlabel('Horizon')
        # ax[2].set_ylabel('Standard Deviation of Elites')

        # ax[0].plot(hist)
        # for i in range(self.n_elite):
        #     ax[1].plot(U_elite[i])
        # ax[2].plot(self.std)
        # fig.tight_layout()
        # fig.savefig('test.png', dpi = 300)
        # plt.show()

        ## step forward mean and std
        self.mean = jnp.roll(self.mean, shift = -1, axis = 0)
        self.std = jnp.roll(self.std, shift = -1, axis = 0)

        self.mean = self.mean.at[-1].set(self.mean[-2])
        self.std = self.std.at[-1].set(self.std[-2])
        # self.mean = self.mean.at[-1].set(jnp.zeros_like(self.mean[-2]))
        # self.std = self.std.at[-1].set(jnp.ones_like(self.std[-2]))

        best_idx = jnp.argmax(J)
        U_best = U_batch[best_idx]
        ut = U_best[0]

        return ut, J[elite_idx].mean()

if __name__ == '__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    
    shots = 1024
    horizon = 1000
    iterations = 1
    elite_frac = 0.1
    smoothing = 0.3
    objective_type = 'mean_neg_cl_ent'
    alpha = 1.0

    # pendulum = 'single'
    pendulum = 'double'

    if pendulum == 'single':
        dt = 0.05
        dyn = Dynamics('xml/pendulum.xml', dt = dt)
        steps = 500
    elif pendulum == 'double':
        dt = 0.01
        dyn = Dynamics('xml/double_pendulum.xml', dt = dt)
        steps = 3000

    step = make_step(dyn)

    mpc = CEM(
        dyn, 
        make_compute_rate(dyn, objective_type, alpha),
        shots, 
        horizon, 
        iterations, 
        elite_frac,
        smoothing)

    theta = 0.0
    x0 = jnp.zeros(dyn.state_dim)
    x0 = x0.at[0].set(theta)
    xt = x0.copy()

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

    name = f'CEM-obj={objective_type}-shots={shots}-h={horizon}-iter={iterations}-frac={elite_frac}-smooth={smoothing}-alpha={alpha}'
    
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    if objective_type == 'rate' or objective_type == 'mean_rate':
        ax.set_ylabel('nats / s')
    elif objective_type == 'mean_info' or objective_type == 'mean_neg_cl_ent':
        ax.set_ylabel('nats')
    T = jnp.arange(0, steps)
    ax.plot(T * dt, hist)
    fig.tight_layout()
    fig.savefig(name + '.png', dpi = 300)
    plt.show()

    if pendulum == 'single':
        dyn.render(X, path = name + '.mp4', skip = 1)
    elif pendulum == 'double':
        dyn.render(X, path = name + '.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))

