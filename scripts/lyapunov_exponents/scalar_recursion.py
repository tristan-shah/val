from jax import numpy as jnp
import matplotlib.pyplot as plt

def scalar_recursion(horizon: int, a: float):

    hist = jnp.zeros(horizon)
    T = jnp.arange(1, horizon + 1)

    y = 1.0
    for t in reversed(range(horizon)):
        y = 1.0 + a**2 * y
        hist = hist.at[t].set(y)
        print(y)

    hist = jnp.log(hist)

    return hist / (2 * jnp.flip(T))


def log_scalar_recursion(horizon: int, a: float):

    hist = jnp.zeros(horizon)
    T = jnp.arange(1, horizon + 1)

    x = jnp.array(0.0) ## x = ln(y)

    for t in reversed(range(horizon)):

        z = 2 * jnp.log(a) + x
        x = jnp.maximum(z, 0) + jnp.log(1.0 + jnp.exp(-jnp.abs(z)))

        hist = hist.at[t].set(x)
        print(x)

    return hist / (2 * jnp.flip(T))

if __name__ == '__main__':

    horizon = 100
    a = jnp.array(3)

    LE = jnp.log(jnp.abs(a))

    unstable_hist = scalar_recursion(horizon, a)
    stable_hist = log_scalar_recursion(horizon, a)

    fig, ax = plt.subplots(1, 1)

    fig.suptitle('Scalar Recursion:\n' + r'$y_t = 1.0 + a^2 y_{t+1}, \qquad a = 3$')
    
    ax.plot(stable_hist, label = 'Stable Recursion', color = 'orange')
    ax.plot(unstable_hist, label = 'Unstable Recursion', color = 'blue', linestyle = 'dashed')
    ax.hlines(LE.clip(min = 0.0), xmin = 0, xmax = horizon, colors = 'r', label = 'True Open-Loop')
    ax.legend()
    plt.show()