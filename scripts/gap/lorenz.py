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

if __name__ == '__main__':

    dt = 0.01
    horizon = 1000

    x0 = jnp.array([1.0, 1.0, 1.0])

    sigma = 16.0
    rho = 45.92
    beta = 4.0


    step = make_lorenz_step(dt, sigma, rho, beta)
    unroll = make_unroll(step)

    linearize = jax.jacfwd(step)
    traj_linearize = jax.vmap(linearize)

    U = jnp.zeros((horizon, 1))
    X = unroll(x0, U)

    fx = traj_linearize(X[:-1], U)
    fu = jnp.zeros((horizon, 3, 1))

    Y, V, W, K = compute_volume(fx, fu, 1.0, 1.0)

    T = jnp.arange(1, horizon + 1) * dt
    ol_entropy = jnp.linalg.slogdet(Y).logabsdet / (2 * jnp.flip(T))

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f'Estimation of KSE for Lorenz (sigma = {sigma}, rho = {rho}, beta = {beta})')
    ax.set_xlabel('Backwards Recursion Step')
    ax.set_ylabel('KSE (bits/iter)')

    ax.plot(ol_entropy)
    
    ax.set_ylim(0.0, 5.0)
    fig.savefig('lorenz.png', dpi = 300)
    plt.show()