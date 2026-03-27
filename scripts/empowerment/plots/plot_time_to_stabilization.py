from jax import numpy as jnp
import matplotlib.pyplot as plt

from val.utils import smooth_angle_wrap

if __name__ == '__main__':


    # ## single pendulum
    # empowerment = jnp.load('/Users/tristanshah/Desktop/code/val/single_pendulum-empowerment-h=150-traj.npy')
    # ol = jnp.load('/Users/tristanshah/Desktop/code/val/results/SINGLE_PENDULUM/exponential_domain/ol/seed=5-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05/traj.npy')

    # empowerment_angle = jnp.abs(smooth_angle_wrap(empowerment[:, 0]))
    # ol_angle = jnp.abs(smooth_angle_wrap(ol[:, 0]))


    # fig, ax = plt.subplots(1, 1)
    # ax.plot(empowerment_angle, label = 'Empowerment')
    # ax.plot(ol_angle, label = 'OL')
    # ax.legend()
    # fig.tight_layout()
    # plt.show()
    

    ## cart pole
    seed = 0
    dt = 0.01
    angle_thresh = 0.3
    empowerment = jnp.load(f'/Users/tristanshah/Desktop/code/val/results/empowerment/CART_POLE/h=300-dt={dt}/traj.npy')
    cip = jnp.load(f'/Users/tristanshah/Desktop/code/val/results/CIP/CART_POLE/exponential_domain/cip/seed={seed}-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt={dt}/traj.npy')
    ol = jnp.load(f'/Users/tristanshah/Desktop/code/val/results/CIP/CART_POLE/exponential_domain/ol/seed={seed}-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt={dt}/traj.npy')
    
    
    empowerment_angle = jnp.abs(smooth_angle_wrap(empowerment[:, 1] - jnp.pi))
    cip_angle = jnp.abs(smooth_angle_wrap(cip[:, 1] - jnp.pi))
    ol_angle = jnp.abs(smooth_angle_wrap(ol[:, 1] - jnp.pi))

    empowerment_at_top = jnp.abs(smooth_angle_wrap(empowerment[:, 1] - jnp.pi)) < angle_thresh
    cip_at_top = jnp.abs(smooth_angle_wrap(cip[:, 1] - jnp.pi)) < angle_thresh
    ol_at_top = jnp.abs(smooth_angle_wrap(ol[:, 1] - jnp.pi)) < angle_thresh

    times = jnp.linspace(0.0, empowerment.shape[0] * dt, empowerment.shape[0])
    fig, ax = plt.subplots(2, 1)
    fig.suptitle('Cart Pole')
    ax[0].set_ylabel('Angle From Top (rad)')
    ax[0].plot(times, empowerment_angle, label = 'Empowerment')
    ax[0].plot(times, cip_angle, label = 'CIP')
    ax[0].plot(times, ol_angle, label = 'OL')
    ax[0].legend()

    ax[1].set_ylabel('Within pi +- 0.2')
    ax[1].plot(times, empowerment_at_top, label = 'Empowerment')
    ax[1].plot(times, cip_at_top, label = 'CIP')
    ax[1].plot(times, ol_at_top, label = 'OL')
    ax[1].legend()
    ax[1].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig('cart_pole_comparison.png', dpi = 300)
    plt.show()




    ## double pendulum
    seed = 0
    dt = 0.01
    empowerment = jnp.load(f'/Users/tristanshah/Desktop/code/val/results/empowerment/DOUBLE_PENDULUM/h=500-dt={dt}/traj.npy')[:1201]
    cip = jnp.load(f'/Users/tristanshah/Desktop/code/val/results/CIP/DOUBLE_PENDULUM/cip/seed={seed}-gear=6.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt={dt}/traj.npy')
    ol = jnp.load(f'/Users/tristanshah/Desktop/code/val/results/CIP/DOUBLE_PENDULUM/ol/seed={seed}-gear=6.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt={dt}/traj.npy')


    angle_thresh = 0.3
    
    
    empowerment_angle_link_1 = jnp.abs(smooth_angle_wrap(empowerment[:, 0] - jnp.pi))
    cip_angle_link_1 = jnp.abs(smooth_angle_wrap(cip[:, 0] - jnp.pi))
    ol_angle_link_1 = jnp.abs(smooth_angle_wrap(ol[:, 0] - jnp.pi))

    empowerment_angle_link_2 = jnp.abs(smooth_angle_wrap(empowerment[:, 0] + empowerment[:, 1] - jnp.pi))
    cip_angle_link_2 = jnp.abs(smooth_angle_wrap(cip[:, 0] + cip[:, 1] - jnp.pi))
    ol_angle_link_2 = jnp.abs(smooth_angle_wrap(ol[:, 0] + ol[:, 1] - jnp.pi))

    empowerment_at_top_link_1 = empowerment_angle_link_1 < angle_thresh
    cip_at_top_link_1 = cip_angle_link_1 < angle_thresh
    ol_at_top_link_1 = ol_angle_link_1 < angle_thresh

    empowerment_at_top_link_2 = empowerment_angle_link_2 < angle_thresh
    cip_at_top_link_2 = cip_angle_link_2 < angle_thresh
    ol_at_top_link_2 = ol_angle_link_2 < angle_thresh

    times = jnp.linspace(0.0, empowerment.shape[0] * dt, empowerment.shape[0])
    fig, ax = plt.subplots(3, 1)
    fig.suptitle('Double Pendulum')
    ax[0].set_ylabel('Link 1 \n Angle From Top \n (rad)')
    ax[0].plot(times, empowerment_angle_link_1, label = 'Empowerment')
    ax[0].plot(times, cip_angle_link_1, label = 'CIP')
    ax[0].plot(times, ol_angle_link_1, label = 'OL')
    ax[0].legend()
    
    ax[1].set_ylabel('Link 2 \n Angle From Top \n (rad)')
    ax[1].plot(times, empowerment_angle_link_2, label = 'Empowerment')
    ax[1].plot(times, cip_angle_link_2, label = 'CIP')
    ax[1].plot(times, ol_angle_link_2, label = 'OL')
    ax[1].legend()


    ax[2].set_ylabel(f'Both Links \n Within pi +- {angle_thresh}')
    ax[2].plot(times, empowerment_at_top_link_1 & empowerment_at_top_link_2, label = 'Empowerment')
    ax[2].plot(times, cip_at_top_link_1 & cip_at_top_link_2, label = 'CIP')
    ax[2].plot(times, ol_at_top_link_1 & ol_at_top_link_2, label = 'OL')
    ax[2].legend()
    
    ax[2].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig('double_pendulum_comparison.png', dpi = 300)
    plt.show()






