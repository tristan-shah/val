import jax 
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt
import control as ct

from val.info import compute_volume

def estimate_lyapunov_exponents(fx: Array):

    horizon = fx.shape[0]
    dx = fx.shape[-1]
    I = jnp.eye(dx)
    T = jnp.arange(1, horizon + 1)

    J = I

    lyapunov_exponents = jnp.zeros(dx)
    hist = jnp.zeros((horizon, dx))

    for t in range(horizon):

        J = fx[t] @ J
        Q, R = jnp.linalg.qr(J)
        lyapunov_exponents += jnp.log(jnp.abs(jnp.diag(R)))
        J = Q

        hist = hist.at[t].set(lyapunov_exponents)

    return hist / T[:, None]

def inverse_method(fx: Array):
    '''
    Woodbury inverse propagation (information filter)
    '''

    horizon = fx.shape[0]
    T = jnp.arange(1, horizon + 1)
    dx = fx.shape[-1]
    I = jnp.eye(dx)

    O = I
    y = jnp.array(0.0)

    hist = jnp.zeros(horizon)

    for t in reversed(range(horizon)):
        y = y + jnp.linalg.slogdet(O + fx[t] @ fx[t].T).logabsdet
        O = I - fx[t].T @ jnp.linalg.inv(O + fx[t] @ fx[t].T) @ fx[t]
        # y = y + jnp.linalg.slogdet(O + fx[t].T @ fx[t]).logabsdet

        # hist = hist.at[t].set(-jnp.linalg.slogdet(O).logabsdet)
        hist = hist.at[t].set(y)

    hist = hist / (2 * jnp.flip(T))

    return hist

def qr_method(fx: Array):
    '''
    Propagate the square root of the recursion using QR decomposition
    '''

    horizon = fx.shape[0]
    T = jnp.arange(1, horizon + 1)
    dx = fx.shape[-1]
    I = jnp.eye(dx)

    ## for QR method
    R = jnp.linalg.cholesky(I)

    hist = jnp.zeros(horizon)

    cond = jnp.zeros(horizon)

    scale = 0.0

    for t in reversed(range(horizon)):

        cond = cond.at[t].set(jnp.linalg.cond(R))

        ## for QR method
        P = jnp.concatenate([
            I,
            R @ fx[t]
        ])

        _, R = jnp.linalg.qr(P)

        '''
        removing exponential growth
        '''
        # alpha = jnp.linalg.slogdet(R).logabsdet / dx
        # R = R / jnp.exp(alpha)
        # scale += dx * alpha

        # print(R)
        # print(jnp.linalg.slogdet(R).logabsdet)
        # exit()

        scale = jnp.linalg.slogdet(R).logabsdet
        hist = hist.at[t].set(scale)

        print(scale)

    hist = hist / jnp.flip(T)

    return hist, cond

def log_sum_exp_calculation(fx: Array):

    horizon = fx.shape[0]
    T = jnp.arange(1, horizon + 1)
    n = fx.shape[-1]
    
    LE = estimate_lyapunov_exponents(fx) * T[:, None]

    '''
    Using only the global max LE results in extremely small cumulative sums
    '''
    # max_LE = jnp.max(LE)
    # normalized_LE = LE - max_LE
    # time_exponential_LE = jnp.exp(normalized_LE)
    # cumulative_LE = jnp.cumsum(time_exponential_LE, axis = 0)
    # l_logsumexp = n * max_LE + jnp.sum(jnp.log(cumulative_LE), axis = -1)
    # return l_logsumexp / T

    '''
    Using the maximum over time for each LE individually
    '''
    max_LE = jnp.max(LE, axis = 0)
    normalized_LE = LE - max_LE[None, :]
    time_exponential_LE = jnp.exp(normalized_LE)
    cumulative_LE = jnp.cumsum(time_exponential_LE, axis = 0)
    l_logsumexp = jnp.sum(max_LE) + jnp.sum(jnp.log(cumulative_LE), axis = -1)
    return l_logsumexp / T

def matrix_log_recursion(fx: Array):
    '''
    Applying the recursion for scalars to matrices
    '''

    horizon = fx.shape[0]
    dx = fx.shape[-1]
    I = jnp.eye(dx)
    T = jnp.arange(1, horizon + 1)
    hist = jnp.zeros(horizon)

    X = jnp.zeros((dx, dx))

    for t in reversed(range(horizon)):

        ## get current jacobian
        A = fx[t]

        '''
        What we are really trying to do in this first part is to compute
        Z = \ln A^T Y A
        '''
        ## first we form Z by doing a matrix log on A.T @ A
        D, Q = jnp.linalg.eigh(A.T @ A) ## correct decomposition: Q @ D @ Q.T = A.T @ A
        drift = Q @ jnp.diag(jnp.log(D + 1e-6)) @ Q.T
        Z = drift + X


        '''
        Attempting a more stable calculation of Z (this actually makes it worse)
        '''
        # Dx, Qx = jnp.linalg.eigh(X)
        # max_eig = jnp.max(Dx)

        # W = A.T @ Qx @ jnp.diag(jnp.exp(Dx - max_eig)) @ Qx.T @ A
        # # Optimized W calculation
        # # V = jnp.exp(0.5 * (Dx - max_eig))[:, None] * (Qx.T @ A)
        # # W = V.T @ V
        # Dw, Qw = jnp.linalg.eigh(W)

        # ## now we form Z
        # Z = max_eig * I + Qw @ jnp.diag(jnp.log(Dw)) @ Qw.T


        ## next we propagate X
        Dz, Qz = jnp.linalg.eigh(Z)
        D_softplus = jnp.maximum(Dz, 0) + jnp.log1p(jnp.exp(-jnp.abs(Dz)))
        X = Qz @ jnp.diag(D_softplus) @ Qz.T

        print(X)
        # Ensure symmetry for the next iteration
        X = 0.5 * (X + X.T)

        hist = hist.at[t].set(jnp.trace(X))

    return hist / (2 * jnp.flip(T))

def switch_recursion(fx: Array):
    
    '''
    Switch to a different recursion: ln(A^T Y A) after Y becomes large
    '''

    horizon = fx.shape[0]
    dx = fx.shape[-1]
    I = jnp.eye(dx)
    T = jnp.arange(1, horizon + 1)
    hist = jnp.zeros(horizon)

    Y = I

    for t in reversed(range(horizon)):

        A = fx[t]

        if t > 260:
            Y = I + A.T @ Y @ A
            logdet_Y = jnp.linalg.slogdet(Y).logabsdet
        else:

            D, Q = jnp.linalg.eigh(A.T @ A)
            # logdet_Y = jnp.sum(jnp.clip(jnp.log(D + 1e-3), min = 0.0)) + logdet_Y
            logdet_Y = jnp.linalg.slogdet(A.T @ A).logabsdet + logdet_Y

        hist = hist.at[t].set(logdet_Y)

    return hist / (2 * jnp.flip(T))

def matrix_softplus_recursion(fx: Array):

    horizon = fx.shape[0]
    dx = fx.shape[-1]
    I = jnp.eye(dx)
    T = jnp.arange(1, horizon + 1)
    hist = jnp.zeros(horizon)

    X = jnp.zeros((dx, dx))

    for t in reversed(range(horizon)):

        Dx, Qx = jnp.linalg.eigh(X)
        max_eig = jnp.max(Dx)

        M = fx[t].T @ Qx @ jnp.diag(jnp.exp(Dx - max_eig)) @ Qx.T @ fx[t]
        Dm, Qm = jnp.linalg.eigh(M)
        print(Dm)

        exit()
    #     W = jnp.exp(-max_eig) * I + M
            
    #     Dw, Qw = jnp.linalg.eigh(W)
    #     X = max_eig * I + Qw @ jnp.diag(jnp.log(Dw)) @ Qw.T

    #     X = 0.5 * (X.T + X)
    #     hist = hist.at[t].set(jnp.trace(X))

    # return hist / (2 * jnp.flip(T))

def max_log_eigenvalue_recursion(fx: Array):

    horizon = fx.shape[0]
    dx = fx.shape[-1]
    T = jnp.arange(1, horizon + 1)
    hist = jnp.zeros(horizon)

    x = jnp.array(0.0)

    for t in reversed(range(horizon)):
        
        A = fx[t]
        a = jnp.max(jnp.linalg.eigvalsh(A.T @ A))
        z = jnp.log(a) + x

        x = jnp.maximum(z, 0) + jnp.log(1.0 + jnp.exp(-jnp.abs(z)))
        hist = hist.at[t].set(x)
        print(x)

    return dx * hist / (2 * jnp.flip(T))

def softplus_intuition_recursion(fx: Array):

    horizon = fx.shape[0]
    dx = fx.shape[-1]
    T = jnp.arange(1, horizon + 1)
    hist = jnp.zeros(horizon)

    X = jnp.zeros((dx, dx))

    for t in reversed(range(horizon)):
        
        A = fx[t]
        Da, Qa = jnp.linalg.eigh(A.T @ A)
        M = Qa @ jnp.diag(jnp.log(Da)) @ Qa.T

        Dx, Qx = jnp.linalg.eigh(M + X)

        # X = Qx @ jnp.diag(jnp.maximum(Dx, 0)) @ Qx.T
        X = Qx @ jnp.diag(jax.nn.softplus(Dx)) @ Qx.T
        print(X)
        hist = hist.at[t].set(jnp.trace(X))

    return hist / (2 * jnp.flip(T))

if __name__ == '__main__':

    horizon = 300
    T = jnp.arange(1, horizon + 1)

    '''
    controllable
    '''
    ## one large negative
    A = jnp.array([
        [0.01, 0, 0],
        [0, 10, 0],
        [0, 0, 5]
    ])

    # ## all positive
    # A = jnp.array([
    #     [10, 1, 0],
    #     [3, 3, 1],
    #     [-1, -2, -5]
    # ])

    # A = jnp.array([
    #     [2, 5, 0],
    #     [3, 3, 0.1],
    #     [-1, -1, -5]
    # ])

    A = jnp.array([
        [0.1, 10, 0],
        [0, 2, 0],
        [0, 0, 3]
    ])

    A = jax.random.normal(jax.random.PRNGKey(88888888), (3, 3)) * 2.0

    B = jnp.array([
        [0],
        [0],
        [1]
    ])

    dx = A.shape[0]

    LE = jnp.log(jnp.abs(jnp.linalg.eigvals(A)))

    fx = jnp.zeros((horizon, dx, dx))
    fu = jnp.zeros((horizon, dx, 1))
    fx = fx.at[:].set(A)
    fu = fu.at[:].set(B)

    # hist = matrix_softplus_recursion(fx)
    # hist = max_log_eigenvalue_recursion(fx)
    # hist = matrix_log_recursion(fx)
    # hist = softplus_intuition_recursion(fx)
    hist = inverse_method(fx)

    Y, _, _, _ = compute_volume(fx, fu, 1.0, 1.0)
    ol_entropy = jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T))
    
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Estimation of KSE for a Linear System with\n LE = {LE}')
    ax.set_xlabel('Backwards Recursion Step')
    ax.set_ylabel('KSE (bits/iter)')
    ax.plot(ol_entropy, label = 'Unstable Open-Loop')
    ax.plot(hist, label = 'Stable? Open-Loop')

    ax.hlines(jnp.sum(LE.clip(min = 0.0)), xmin = 0, xmax = horizon, colors = 'r', linestyles = 'dashed', label = 'True Open-Loop')

    ax.legend(title = 'KSE')
    fig.tight_layout()
    fig.savefig('not_all_positive_LE.png', dpi = 300)
    plt.show()