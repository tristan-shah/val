from jax import numpy as jnp
import matplotlib.pyplot as plt

if __name__ == '__main__':

    plt.rcParams.update({
        'font.size':        7,
        'axes.titlesize':   8,
        'axes.labelsize':   7,
        'xtick.labelsize':  6,
        'ytick.labelsize':  6,
        'legend.fontsize':  6,
        'lines.linewidth':  1.0,
    })

    dt = 0.01

    cart_pole_ol       = jnp.load('results/CIP/CART_POLE/ol/seed=0-beta=0.0-h=512-shots=1024-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    double_pendulum_ol = jnp.load('results/CIP/DOUBLE_PENDULUM/ol/seed=0-beta=0.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    triple_pendulum_ol = jnp.load('results/CIP/TRIPLE_PENDULUM/ol/retest-seed=0-gear=25.0-beta=3.0-h=512-shots=2048-iter=3-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    humulum_ol         = jnp.load('results/CIP/HUMULUM/efficient/ol/seed=1-beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]

    T = humulum_ol.shape[0]
    t = jnp.linspace(0.0, T * dt, T)

    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.6))

    ax.plot(t, cart_pole_ol,       label='Cart pole')
    ax.plot(t, double_pendulum_ol, label='Double pendulum')
    ax.plot(t, triple_pendulum_ol, label='Triple pendulum')
    ax.plot(t, humulum_ol,         label='Gibbon')

    ax.set_xlim(0.0, 12.0)
    ax.set_xticks([0, 6, 12])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nats/s')

    fig.tight_layout(pad=0.5, rect=[0, 0.11, 1, 1])
    fig.legend(loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=6)
    fig.savefig('rate.pdf')
    fig.savefig('rate.png', dpi=600)
    plt.show()
