import jax 
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import make_unroll
from val.info import compute_volume

def make_lorenz_step(dt: float, sigma=16.0, rho=45.92, beta=4.0):

    def step(state: Array, control: Array):

        x, y, z = state

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        return jnp.array([
            x + dt * dx,
            y + dt * dy,
            z + dt * dz
        ])
    
    return jax.jit(step)

def plot_lorenz(X):
    """
    X: Array of shape (T, 3)
    """
    X = jnp.asarray(X)

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    # --- 3D phase plot ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    ax.plot(x, y, z, lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz Attractor")

    plt.tight_layout()
    plt.show()

@jax.jit
def compute_ol(fx: Array):
    
    dx = fx.shape[-1]
    du = fu.shape[-1]

    Q = jnp.eye(dx)

    def scan_fn(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        
        ol_t = carry
        fx_t = inputs

        ol_t = ol_t + jnp.linalg.slogdet(Q + fx_t.T @ fx_t).logabsdet

        carry = ol_t

        return carry, carry
    
    _, ol = jax.lax.scan(scan_fn, init = jnp.linalg.slogdet(Q).logabsdet, xs = fx, reverse = True)
    return ol

if __name__ == '__main__':

    dt = 0.01
    horizon = 100 #40

    # A = jnp.array([
    #     [3.0, 2.0],
    #     [1.0, 0.1]
    # ])


    A = jnp.array([
        [0.1, 0.0, 0.1],
        [1.0, 0.1, 2.0],
        [1.0, 2.0, 0.4]
    ])


    LE = jnp.log(jnp.abs(jnp.linalg.eigvals(A)))
    print(LE)

    dx = A.shape[0]

    fx = jnp.zeros((horizon, dx, dx))
    fu = jnp.zeros((horizon, dx, 1))
    fx = fx.at[:].set(A)

    Y, V, W, K = compute_volume(fx, fu, 1.0, 1.0)

    T = jnp.arange(1, horizon + 1)
    ol_entropy = jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T))
    ol = compute_ol(fx) / (2 * jnp.flip(T))

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Estimation of KSE for Linear System')
    ax.set_xlabel('Backwards Recursion Step')
    ax.set_ylabel('KSE (bits/iter)')
    # ax.set_ylim(0.0, jnp.max(ol) * 1.2)
    ax.plot(ol_entropy, label = 'Estimated')
    # ax.plot(ol, label = 'Estimated')
    ax.hlines(jnp.sum(LE.clip(min = 0.0)), xmin = 0, xmax = horizon, colors = 'r', linestyles = 'dashed', label = 'True KSE')

    ax.legend()
    fig.tight_layout()
    fig.savefig('KSE.png', dpi = 300)
    plt.show()