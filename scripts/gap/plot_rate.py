from jax import numpy as jnp
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dt = 0.05

    h_50 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=50.npy')
    h_100 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=100.npy')
    h_200 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=200.npy')
    h_300 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=300.npy')

    T = jnp.arange(0, 600) * dt

    fig, ax = plt.subplots(2, 1, figsize = (9, 8))
    
    fig.suptitle('Online MPC Chaos Consumption (Single Pendulum)')

    ax[0].set_title(r'Information: $\ln\det Y - \ln \det V$')
    ax[0].set_ylabel('nats')

    ax[0].plot(T, h_50 * 50 * dt, label = '50')
    ax[0].plot(T, h_100 * 100 * dt, label = '100')
    ax[0].plot(T, h_200 * 200 * dt, label = '200')
    ax[0].plot(T, h_300 * 300 * dt, label = '300')
    ax[0].legend(title = 'Horizon')

    ax[1].set_title(r'Information Rate: $\frac{1}{T}(\ln\det Y - \ln \det V)$')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('nats / s')
    ax[1].plot(T, h_50, label = '50')
    ax[1].plot(T, h_100, label = '100')
    ax[1].plot(T, h_200, label = '200')
    ax[1].plot(T, h_300, label = '300')
    ax[1].legend(title = 'Horizon')

    fig.tight_layout()
    fig.savefig('rate.png', dpi = 300)
    plt.show()