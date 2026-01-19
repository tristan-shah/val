from jax import numpy as jnp
import matplotlib.pyplot as plt

from val.utils import smooth_angle_wrap

if __name__ == '__main__':

    dt = 0.05

    horizons = jnp.arange(150, 325, 25)

    T = jnp.arange(0, 600) * dt
    T = T[:350]


    # '''
    # '''
    # fig, ax = plt.subplots(2, 1)

    # ax[0].set_title('Controllable Information Production')
    # ax[0].set_ylabel('Absolute Angle From Top (rad)')

    # ax[1].set_title('Empowerment')
    # ax[1].set_ylabel('Absolute Angle From Top (rad)')
    # ax[1].set_xlabel('Environment Time (s)')

    # for h in horizons:

    #     X_cip = jnp.load(f'CEM-single_pendulum/h={h}-gamma=1.0-shots=512-iter=1-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt={dt}-traj.npy')
    #     X_empowerment = jnp.load(f'EMPOWERMENT-single_pendulum/empowerment-h={h}-traj.npy')

    #     X_cip = X_cip[:350]
    #     X_empowerment = X_empowerment[:350]

    #     cip_angle_from_top = jnp.abs(smooth_angle_wrap(X_cip[:, 0] - jnp.pi))
    #     empowerment_angle_from_top = jnp.abs(smooth_angle_wrap(X_empowerment[:, 0] - jnp.pi))

    #     ax[0].plot(T, cip_angle_from_top, label = h)
    #     ax[1].plot(T, empowerment_angle_from_top)

    # fig.legend(title = 'Horizon')
    # fig.tight_layout()
    # fig.savefig('single_pendulum.png', dpi = 400)
    # plt.show()


    '''
    '''
    fig, ax = plt.subplots(2, 1)
    
    fig.suptitle('MPC on Single Pendulum')

    ax[0].set_ylabel('Information Production (nats/s)')
    ax[0].set_ylim(0.0, 1.5)

    ax[1].set_ylabel('Absolute Angle From Top (rad)')
    ax[1].set_xlabel('Environment Time (s)')

    for h in horizons:

        hist = 0.5 * jnp.load(f'CEM-single_pendulum/h={h}-gamma=1.0-shots=512-iter=1-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt={dt}-hist.npy')
        X = jnp.load(f'CEM-single_pendulum/h={h}-gamma=1.0-shots=512-iter=1-elite=0.1_keep=0.3-smooth=0.1-alpha=1.0-dt={dt}-traj.npy')

        hist = hist[:350]
        X = X[:350]

        angle_from_top = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))

        ax[0].plot(T, hist, label = h)
        ax[1].plot(T, angle_from_top)

    fig.legend(title = 'Horizon')
    fig.tight_layout()
    fig.savefig('single_pendulum.png', dpi = 400)
    plt.show()
