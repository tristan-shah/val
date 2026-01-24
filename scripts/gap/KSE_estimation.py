import jax 
from jax import Array
from jax import numpy as jnp
from einops import einsum
import matplotlib.pyplot as plt

from val import make_unroll
from val.info import compute_volume
from val.utils import estimate_lyapunov_hist

@jax.jit
def compute_inverse(fx: Array, fu: Array, alpha: float, gamma: float):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx) * 1.0
    R = jnp.eye(du) * alpha

    def scan_fn(carry: tuple[Array, Array, Array], inputs: tuple[Array, Array]):
        
        Y_t, V_t, W_t = carry
        fx_t, fu_t = inputs

        Y_t = 0.5 * (Y_t + Y_t.T)

        S_inv = jnp.linalg.inv(R + gamma * fu_t.T @ V_t @ fu_t)
        ## feedback gain
        K_t = - gamma * S_inv @ fu_t.T @ V_t @ fx_t
        ## open loop entropy
        Y_t = Q + gamma * fx_t.T @ Y_t @ fx_t
        # Y_t = Q - fx_t.T @ jnp.linalg.inv(Y_t + fx_t @ fx_t.T) @ fx_t
        ## riccati equation
        V_t = Q + gamma * fx_t.T @ V_t @ fx_t - gamma ** 2 * fx_t.T @ V_t @ fu_t @ S_inv @ fu_t.T @ V_t @ fx_t
        ## closed loop entropy
        W_t = Q + gamma * (fx_t + fu_t @ K_t).T @ W_t @ (fx_t + fu_t @ K_t)

        carry = (Y_t, V_t, W_t)

        return carry, (Y_t, V_t, W_t, K_t)
    
    _, (Y, V, W, K) = jax.lax.scan(scan_fn, init = (Q, Q, Q), xs = (fx, fu), reverse = True)
    return Y, V, W, K


# @jax.jit
# def compute_log_eigen_qr(fx: Array, gamma: float = 1.0):
#     """
#     Proper stable log-eigenvalue recursion for Y_t = Q + gamma F_t^T Y_{t+1} F_t
#     using full square-root QR decomposition.
#     """
#     T, dx, _ = fx.shape
#     S_next = jnp.linalg.cholesky(jnp.eye(dx))  # terminal S_T = sqrt(Q)

#     def scan_fn(S_next, F_t):
#         # stack for QR
#         A = jnp.vstack([jnp.linalg.cholesky(jnp.eye(dx)), jnp.sqrt(gamma) * (S_next @ F_t)])
#         _, R = jnp.linalg.qr(A, mode='reduced')
#         S_t = R
#         # log eigenvalues of Y_t
#         log_eigs_Y_t = 2 * jnp.log(jnp.abs(jnp.linalg.svd(S_t, compute_uv=False)))
#         return S_t, log_eigs_Y_t

#     _, log_eigs = jax.lax.scan(scan_fn, init=S_next, xs=fx, reverse=True)
#     return log_eigs

if __name__ == '__main__':

    dt = 0.01
    horizon = 100 #40
    T = jnp.arange(1, horizon + 1)

    A = jnp.array([
        [2.0, 0.0, -1.0],
        [5.0, 10.0, 2.0],
        [1.0, 0.5, 5.0]
    ])

    B = jnp.array([
        [-0.1],
        [0.0],
        [0.001]
    ])


    LE = jnp.log(jnp.abs(jnp.linalg.eigvals(A)))

    dx = A.shape[0]

    fx = jnp.zeros((horizon, dx, dx))
    fu = jnp.zeros((horizon, dx, 1))
    fx = fx.at[:].set(A)
    fu = fu.at[:].set(B)


    Y, V, W, K = compute_inverse(fx, fu, 1.0, 1.0)

    D = fx + einsum(fu, K, 'b x1 u, b u x2 -> b x1 x2')

    kse_ol = estimate_lyapunov_hist(fx, 1.0).clip(min = 0.0).sum(axis = 1)
    kse_cl = estimate_lyapunov_hist(D, 1.0).clip(min = 0.0).sum(axis = 1)


    # ol_entropy = -jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T))
    ol_entropy = jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T))
    cl_entropy = jnp.linalg.slogdet(W).logabsdet / (2 * jnp.flip(T))

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Estimation of KSE for Linear System')
    ax.set_xlabel('Backwards Recursion Step')
    ax.set_ylabel('KSE (bits/iter)')
    ax.plot(ol_entropy, label = 'Estimated Open-Loop')
    ax.plot(cl_entropy, label = 'Estimated Closed-Loop')
    ax.plot(kse_ol, label = 'Gram-Shmidt OL')
    ax.plot(kse_cl, label = 'Gram-Shmidt CL')
    # ax.plot(eigs.sum(axis = 1) / (2 * jnp.flip(T)))
    
    ax.hlines(jnp.sum(LE.clip(min = 0.0)), xmin = 0, xmax = horizon, colors = 'r', linestyles = 'dashed', label = 'True Open-Loop')

    ax.legend(title = 'KSE')
    fig.tight_layout()
    fig.savefig('KSE.png', dpi = 300)
    plt.show()