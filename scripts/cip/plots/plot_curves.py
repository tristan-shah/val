from jax import numpy as jnp
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dt = 0.01

    cart_pole_ol = jnp.load('results/CIP/CART_POLE/ol/seed=0-beta=0.0-h=512-shots=1024-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    double_pendulum_ol = jnp.load('results/CIP/DOUBLE_PENDULUM/ol/seed=0-beta=0.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    triple_pendulum_ol = jnp.load('results/CIP/TRIPLE_PENDULUM/ol/retest-seed=0-gear=25.0-beta=3.0-h=512-shots=2048-iter=3-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    humulum_ol = jnp.load('results/CIP/HUMULUM/efficient/ol/seed=1-beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]

    T = humulum_ol.shape[0]
    t = jnp.linspace(0.0, T * dt, T)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0.0, 12.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats/s')

    ax.plot(t, cart_pole_ol, label = 'Cart Pole')
    ax.plot(t, double_pendulum_ol, label = 'Double Pendulum')
    ax.plot(t, triple_pendulum_ol, label = 'Triple Pendulum')
    ax.plot(t, humulum_ol, label = 'Gibbon')

    ax.legend()
    fig.tight_layout()
    fig.savefig('rate.png', dpi = 300)
    plt.show()
