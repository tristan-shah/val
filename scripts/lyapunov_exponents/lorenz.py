import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from val import make_unroll

def make_lorenz_f(sigma: float = 10, rho: float = 28, beta: float = 8/3):
        
    def f(xt: Array, ut: Array):
        dx = sigma * (xt[1] - xt[0])
        dy = xt[0] * (rho - xt[2]) - xt[1]
        dz = xt[0] * xt[1] - beta * xt[2]
        return jnp.stack([dx, dy, dz])
    
    return jax.jit(f)

def make_rk4_step(f: callable, dt: float):
    '''
    Wrap a continuous-time dynamics function f(x, u) into an RK4 integrator step.
    '''

    def step(xt: Array, ut: Array):

        k1 = f(xt, ut)
        k2 = f(xt + 0.5 * dt * k1, ut)
        k3 = f(xt + 0.5 * dt * k2, ut)
        k4 = f(xt + dt * k3, ut)

        xt = xt + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return xt

    return jax.jit(step)

def plot_lorenz_3d(X):
    X = np.asarray(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=0.8)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lorenz attractor')

    plt.show()

@jax.jit
def estimate_lyapunov_exponents(fx: Array, dt: float):
    '''
    fx: (T, d, d) sequence of Jacobians
    returns: (T, d) running Lyapunov exponent estimates
    '''

    horizon, d, _ = fx.shape
    I = jnp.eye(d)

    def step(carry, fx_t):
        J, acc = carry

        J = fx_t @ J
        Q, R = jnp.linalg.qr(J)
        acc = acc + jnp.log(jnp.abs(jnp.diag(R)))
        J = Q

        return (J, acc), acc

    (_, hist), hist = jax.lax.scan(step, (I, jnp.zeros(d)), fx)

    T = jnp.arange(1, horizon + 1)
    return hist / (T[:, None] * dt)

def qr_method(fx: Array, dt: float):

    horizon = fx.shape[0]
    T = jnp.arange(1, horizon + 1)
    dx = fx.shape[-1]
    I = jnp.eye(dx)

    ## for QR method
    Q = I
    R = jnp.linalg.cholesky(I)

    hist = jnp.zeros(horizon)

    for t in reversed(range(horizon)):
        ## for QR method
        P = jnp.concatenate([
            I,
            R @ fx[t]
        ])

        _, R = jnp.linalg.qr(P)

        scale = jnp.linalg.slogdet(R).logabsdet
        hist = hist.at[t].set(scale)

        print(scale)
    hist = hist / (jnp.flip(T) * dt)

    return hist

def max_log_eigenvalue_recursion(fx: Array, dt: float):

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

    return dx * hist / (2 * jnp.flip(T) * dt)

def inverse_method(fx: Array, dt: float):
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
        hist = hist.at[t].set(y)

    hist = hist / (2 * jnp.flip(T) * dt)

    return hist

if __name__ == '__main__':

    horizon = 50000
    dt = 0.02
    
    step = make_rk4_step(make_lorenz_f(), dt = dt)
    unroll = make_unroll(step)
    linearize = jax.vmap(jax.jacfwd(step))

    ## warmstarted state
    xt = jnp.array([-1.00521349e+01, -1.45800861e+01,  2.23401374e+01])
    U = jnp.zeros((horizon, 1))
    X = unroll(xt, U)

    fx = linearize(X[:-1], U)


    hist = inverse_method(fx, dt)
    print(hist)

    LE = estimate_lyapunov_exponents(fx, dt)
    print(LE)

    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(0.0, 1.4)
    ax.set_xlabel('Iteration Timestep')
    ax.set_ylabel('KSE')
    ax.set_title('Lorenz Attractor\nSum of Positve Lyapunov Exponents')
    
    ax.plot(LE.clip(min = 0.0).sum(axis = 1), label = 'Gram Schmidt LE')
    ax.plot(jnp.flip(hist), label = 'CIP Log Domain')

    ax.legend()
    fig.savefig('Lorenz.png', dpi = 300)
    plt.show()

    # plot_lorenz_3d(X)