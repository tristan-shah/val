from jax import numpy as jnp
import matplotlib.pyplot as plt

from val.utils import smooth_angle_wrap

if __name__ == '__main__':

    dt = 0.05

    horizons = jnp.arange(150, 325, 25)

    T = jnp.arange(0, 600) * dt

    # fig, ax = plt.subplots(2, 1, figsize = (9, 8))
    fig, ax = plt.subplots(2, 1)
    
    fig.suptitle('MPC on Single Pendulum')

    ax[0].set_ylabel('Information Production (nats/s)')
    ax[0].set_ylim(0.0, 1.5)

    ax[1].set_ylabel('Absolute Angle From Top (rad)')
    ax[1].set_xlabel('Environment Time (s)')

    for h in horizons:

        hist = 0.5 * jnp.load(f'h={h}-gamma=1.0-shots=512-iter=1-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt={dt}-hist.npy')
        X = jnp.load(f'h={h}-gamma=1.0-shots=512-iter=1-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt={dt}-traj.npy')

        hist = hist[:350]
        X = X[:350]
        T = T[:350]

        angle_from_top = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))

        ax[0].plot(T, hist, label = h)
        ax[1].plot(T, angle_from_top)

    fig.legend(title = 'Horizon')
    fig.tight_layout()
    fig.savefig('single_pendulum.png', dpi = 400)
    plt.show()

    # dt = 0.05
    # h_50 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=50.npy')
    # h_100 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=100.npy')
    # h_200 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=200.npy')
    # h_300 = jnp.load(f'single_pendulum-iterations=5-dt={dt}-closed_loop-h=300.npy')

    # T = jnp.arange(0, 600) * dt

    # fig, ax = plt.subplots(2, 1, figsize = (9, 8))
    
    # fig.suptitle('Online MPC Chaos Consumption (Single Pendulum)')

    # ax[0].set_title(r'Information: $\ln\det Y - \ln \det V$')
    # ax[0].set_ylabel('nats')

    # ax[0].plot(T, h_50 * 50 * dt, label = '50')
    # ax[0].plot(T, h_100 * 100 * dt, label = '100')
    # ax[0].plot(T, h_200 * 200 * dt, label = '200')
    # ax[0].plot(T, h_300 * 300 * dt, label = '300')
    # ax[0].legend(title = 'Horizon')

    # ax[1].set_title(r'Information Rate: $\frac{1}{T}(\ln\det Y - \ln \det V)$')
    # ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('nats / s')
    # ax[1].plot(T, h_50, label = '50')
    # ax[1].plot(T, h_100, label = '100')
    # ax[1].plot(T, h_200, label = '200')
    # ax[1].plot(T, h_300, label = '300')
    # ax[1].legend(title = 'Horizon')

    # fig.tight_layout()
    # fig.savefig('rate.png', dpi = 300)
    # plt.show()