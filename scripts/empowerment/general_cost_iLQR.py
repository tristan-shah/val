import jax
from jax import numpy as jnp
from jax import Array
from einops import einsum
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll
from val.utils import smooth_angle_wrap
from val.empowerment import waterfilling_implicit, compute_power

def make_compute_pendulum_state_cost(Q: Array, smoothing: float = 0.01):

    def compute_pendulum_state_cost(x: Array):
        r = jnp.stack([ smooth_angle_wrap(x[0] - jnp.pi), x[1] ])
        return jnp.sqrt(r.T @ Q @ r + smoothing**2) - smoothing
    
    return jax.jit(compute_pendulum_state_cost)

def make_compute_pendulum_control_cost(limit: float = 0.5):

    def compute_pendulum_control_cost(u: Array):
        return jnp.sum(limit ** 2 * (jnp.cosh(u / limit) - 1.0))

    return jax.jit(compute_pendulum_control_cost)

def make_forward(step: callable, low: Array, high: Array):

    def forward(X: Array, U: Array, k: Array, K: Array, alpha: float):
        
        T = k.shape[0]

        def scan_fn(xt, t):
            delta_x = xt - X[t]
            ut = U[t] + alpha * k[t] + K[t] @ delta_x
            ut = jnp.clip(ut, low, high)
            xt_next = step(xt, ut)
            return xt_next, (xt_next, ut)

        x0 = X[0]
        _, (X_new, U_new) = jax.lax.scan(scan_fn, x0, jnp.arange(T))

        X_new = jnp.concatenate([x0[None, :], X_new])
        return X_new, U_new

    return jax.jit(forward)

def make_backward(traj_linearize: callable, traj_qx: callable, traj_qxx: callable, traj_ru: callable, traj_ruu: callable):
    
    def backward(X: Array, U: Array):

        ## array shapes
        T = U.shape[0]
        dx = X.shape[-1]
        I = jnp.eye(dx)

        ## linearize dynamics
        fx, fu = traj_linearize(X[:-1], U)

        ## state cost gradients
        qx = traj_qx(X)
        qxx = traj_qxx(X)

        ## control cost
        ru = traj_ru(U)
        ruu = traj_ruu(U)

        P = 1.0

        def scan_fn(carry: tuple, t: int):

            Vx, Vxx, L = carry

            # eigs = jnp.linalg.eigvalsh(fu[t].T @ Vxx @ fu[t])#.clip(min = 1e-12)
            eigs = jnp.linalg.eigvalsh(fu[t].T @ L @ fu[t]).clip(min = 1e-12)
            # eigs = jnp.linalg.eigvalsh(fu[t].T @ F.T @ F @ fu[t]).clip(min = 1e-12)
            v = waterfilling_implicit(eigs, P)
            p = compute_power(v, eigs)
            e = 0.5 * jnp.sum(jnp.log(1 + p * eigs))

            ## compute inverse term
            S = jnp.linalg.inv(ruu[t] + fu[t].T @ Vxx @ fu[t])

            ## compute gains
            k = - S @ (fu[t].T @ Vx + ru[t]) ## feedforward
            K = - S @ fu[t].T @ Vxx @ fx[t] ## feedback

            ## total derivative of dynamics
            D = fx[t] + fu[t] @ K
            
            ## value gradient update
            Vx = D.T @ Vx + qx[t] + K.T @ ru[t]
            ## value hessian update
            Vxx = qxx[t] + fx[t].T @ Vxx @ fx[t] - fx[t].T @ Vxx @ fu[t] @ S @ fu[t].T @ Vxx @ fx[t]
            ## force symmetry
            Vxx = 0.5 * (Vxx + Vxx.T)

            # L = D.T @ L @ D + I
            L = D.T @ L @ D + qxx[t]
            # F = F @ D

            carry = (Vx, Vxx, L)

            return carry, (k, K, e)
        
        ## value gradient and hessian
        Vx_T = qx[-1]
        Vxx_T = qxx[-1]
        L_T = qxx[-1]
        # F_T = I
        init = (Vx_T, Vxx_T, L_T)
        
        _, (k, K, e) = jax.lax.scan(scan_fn, init, jnp.arange(T), reverse = True)
        return k, K, e
    
    return jax.jit(backward)

class iLQR:
    def __init__(self, dyn: Dynamics, state_cost: callable, control_cost: callable):

        ## line search parameters
        self.alphas = jnp.linspace(0.05, 5.0, 30)
        # self.alphas = jnp.linspace(0.0001, 5.0, 60)

        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

        self.dx = dyn.state_dim
        self.du = dyn.control_dim

        ## dynamics helper functions
        self.step = make_step(dyn)
        self.unroll = make_unroll(self.step)
        self.traj_linearize = jax.jit(jax.vmap(jax.jacfwd(self.step, argnums = (0, 1))))

        ## for unrolling dynamics with control gains (after backward pass)
        self.forward = make_forward(self.step, self.low, self.high)
        self.batch_forward = jax.jit(jax.vmap(self.forward, in_axes = (None, None, None, None, 0)))

        ## costs

        ## state cost
        q = state_cost
        qx = jax.grad(q)
        qxx = jax.jacfwd(qx)
        self.traj_q = jax.jit(jax.vmap(q))
        self.traj_qx = jax.jit(jax.vmap(qx))
        self.traj_qxx = jax.jit(jax.vmap(qxx))

        ## control cost
        r = control_cost
        ru = jax.grad(r)
        ruu = jax.jacfwd(ru)
        self.traj_r = jax.jit(jax.vmap(r))
        self.traj_ru = jax.jit(jax.vmap(ru))
        self.traj_ruu = jax.jit(jax.vmap(ruu))

        ## for computing cost of trajectories and batches of trajectories
        self.traj_cost = lambda _X, _U: jnp.sum(self.traj_q(_X)) + jnp.sum(self.traj_r(_U))
        self.batch_traj_cost = jax.vmap(self.traj_cost)

        ## make the backward method
        self.backward = make_backward(self.traj_linearize, self.traj_qx, self.traj_qxx, self.traj_ru, self.traj_ruu)
    
    def __call__(self, X: Array, U: Array):

        ## compute backward pass along nominal trajectory
        k, K, e = self.backward(X, U)
        ## find optimal alpha improvement
        X_batch, U_batch = self.batch_forward(X, U, k, K, self.alphas)

        ## evaluate every forward pass
        J_batch = self.batch_traj_cost(X_batch, U_batch)
        
        idx = jnp.argmin(J_batch)
        X = X_batch[idx]
        U = U_batch[idx]
        J = J_batch[idx]
        alpha = self.alphas[idx]
        return X, U, J, alpha, e

if __name__ == '__main__':

    key = jax.random.PRNGKey(0)
    integrator = 'euler'
    dt = 0.01
    horizon = 900

    ## cost parameters
    # angle_penalty = 20.0
    # velocity_penalty = 0.5

    angle_penalty = 2.0
    velocity_penalty = 0.1


    ## instantiate dynamics
    dyn = Dynamics(path = 'xml/pendulum.xml', integrator = integrator, dt = dt)

    ## building Q
    Q = jnp.array([
        [angle_penalty, 0.0],
        [0.0, velocity_penalty]
    ])

    state_cost = make_compute_pendulum_state_cost(Q)
    control_cost = make_compute_pendulum_control_cost()

    ## instantiate iLQR
    ilqr = iLQR(dyn, state_cost, control_cost)

    xt = jnp.zeros(dyn.state_dim)
    theta_0 = 0.0
    # theta_0 = 2.0
    # theta_0 = 3.0

    xt = xt.at[0].set(theta_0)

    noise = 0.05
    U = (jax.random.normal(key, (horizon, dyn.control_dim)) * noise).clip(ilqr.low, ilqr.high)

    ## unroll initial trajectory
    X = ilqr.unroll(xt, U)
    
    cost = []
    iterations = 300

    e_hist = []

    for i in range(iterations):
        
        ## compute backward pass along nominal trajectory
        k, K, e = ilqr.backward(X, U)
        e_hist.append(e)

        ## find optimal alpha improvement
        X_batch, U_batch = ilqr.batch_forward(X, U, k, K, ilqr.alphas)

        ## evaluate every forward pass
        J_batch = ilqr.batch_traj_cost(X_batch, U_batch)
        ## select best performing forward pass
        idx = jnp.argmin(J_batch)

        X = X_batch[idx]
        U = U_batch[idx]
        J = J_batch[idx]
        print(i, J, ilqr.alphas[idx], jnp.sum(e), e[0])
        cost.append(J)

    e_hist = jnp.stack(e_hist)

    print(e_hist.shape)

    ## plotting cost and controls
    fig, ax = plt.subplots(1, 4, figsize = (15, 5))

    ax[0].set_title('Control Signal')
    ax[0].set_xlabel('Timestep')
    for i in range(U.shape[1]):
        ax[0].plot(U[:, i])

    ax[1].set_title('Trajectory Cost')
    ax[1].set_xlabel('Iterations')
    ax[1].plot(cost)

    ax[2].set_title('Sum of sum-empowerments')
    ax[2].set_xlabel('Iterations')
    ax[2].plot(e_hist.sum(axis = -1))

    ax[3].set_title('Sum-empowerment at different timesteps')
    ax[3].plot(e_hist[:, 0], label = 't = 0')
    ax[3].plot(e_hist[:, 150], label = 't = 150')
    # ax[3].plot(e_hist[:, 200])

    fig.tight_layout()
    fig.savefig('test.png', dpi = 300)
    plt.show()

    dyn.render(X, path = 'test.mp4', skip = 4)