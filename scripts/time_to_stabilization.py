from pathlib import Path

from jax import numpy as jnp
import matplotlib.pyplot as plt

from val.utils import smooth_angle_wrap

def get_single_pendulum_tts(X, thresh, dt):
    angle = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    at_top = angle < thresh
    tts = jnp.argmax(at_top) * dt
    return tts

def get_cart_pole_tts(X, thresh, dt):
    angle = jnp.abs(smooth_angle_wrap(X[:, 1] - jnp.pi))
    at_top = angle < thresh
    tts = jnp.argmax(at_top) * dt
    return tts

def get_double_pendulum_tts(X, thresh, dt):
    angle_link_1 = jnp.abs(smooth_angle_wrap(X[:, 0] - jnp.pi))
    angle_link_2 = jnp.abs(smooth_angle_wrap(X[:, 0] + X[:, 1] - jnp.pi))
    at_top_link_1 = angle_link_1 < thresh
    at_top_link_2 = angle_link_2 < thresh
    at_top = at_top_link_1 & at_top_link_2
    tts = jnp.argmax(at_top) * dt
    return tts

if __name__ == '__main__':

    num_seeds = 6

    ## single pendulum dt is different than the others
    single_pendulum_dt = 0.05
    dt = 0.01
    angle_thresh = 0.3

    root = Path('/Users/tristanshah/Desktop/code/val/results')

    single_pendulum_empowerment = jnp.load(root / f'empowerment/SINGLE_PENDULUM/h=150-dt={single_pendulum_dt}/traj.npy')
    cart_pole_empowerment = jnp.load(root / f'empowerment/CART_POLE/h=300-dt={dt}/traj.npy')
    double_pendulum_empowerment = jnp.load(root / f'empowerment/DOUBLE_PENDULUM/h=500-dt={dt}/traj.npy')

    averages = {
        'single_pendulum_cip_tts': [],
        'single_pendulum_ol_tts': [],
        'cart_pole_cip_tts': [],
        'cart_pole_ol_tts': [],
        'double_pendulum_cip_tts': [],
        'double_pendulum_ol_tts': [],
    }

    for seed in range(num_seeds):

        ## single pendulum
        single_pendulum_cip = jnp.load(root / f'CIP/SINGLE_PENDULUM/exponential_domain/cip/seed={seed}-h=150-shots=512-iter=5-elite=0.1-smooth=0.1-rho=0.9-dt={single_pendulum_dt}/traj.npy')
        single_pendulum_ol = jnp.load(root / f'CIP/SINGLE_PENDULUM/exponential_domain/ol/seed={seed}-h=150-shots=512-iter=5-elite=0.1-smooth=0.1-rho=0.9-dt={single_pendulum_dt}/traj.npy')
        single_pendulum_cip_tts = get_single_pendulum_tts(single_pendulum_cip, angle_thresh, single_pendulum_dt)
        single_pendulum_ol_tts = get_single_pendulum_tts(single_pendulum_ol, angle_thresh, single_pendulum_dt)
        averages['single_pendulum_cip_tts'].append(single_pendulum_cip_tts)
        averages['single_pendulum_ol_tts'].append(single_pendulum_ol_tts)

        ## cart pole
        cart_pole_cip = jnp.load(root / f'CIP/CART_POLE/exponential_domain/cip/seed={seed}-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt={dt}/traj.npy')
        cart_pole_ol = jnp.load(root / f'CIP/CART_POLE/exponential_domain/ol/seed={seed}-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt={dt}/traj.npy')
        cart_pole_cip_tts = get_cart_pole_tts(cart_pole_cip, angle_thresh, dt)
        cart_pole_ol_tts = get_cart_pole_tts(cart_pole_ol, angle_thresh, dt)

        averages['cart_pole_cip_tts'].append(cart_pole_cip_tts)
        averages['cart_pole_ol_tts'].append(cart_pole_ol_tts)

        # double_pendulum_cip = jnp.load(root / f'CIP/DOUBLE_PENDULUM/exponential_domain/cip/seed={seed}-gear=6.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01/traj.npy')
        # double_pendulum_ol = jnp.load(root / f'CIP/DOUBLE_PENDULUM/exponential_domain/ol/seed={seed}-gear=6.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01/traj.npy')
        double_pendulum_cip = jnp.load(root / f'CIP/DOUBLE_PENDULUM/cip/seed={0}-gear=6.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01/traj.npy')
        double_pendulum_ol = jnp.load(root / f'CIP/DOUBLE_PENDULUM/ol/seed={0}-gear=6.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01/traj.npy')
        double_pendulum_cip_tts = get_double_pendulum_tts(double_pendulum_cip, angle_thresh, dt)
        double_pendulum_ol_tts = get_double_pendulum_tts(double_pendulum_ol, angle_thresh, dt)

        averages['double_pendulum_cip_tts'].append(double_pendulum_cip_tts)
        averages['double_pendulum_ol_tts'].append(double_pendulum_ol_tts)





    print('Single Pendulum Time to Stabilization:')
    print(f'Empowerment: {get_single_pendulum_tts(single_pendulum_empowerment, angle_thresh, single_pendulum_dt)}')
    print(f"CIP averaged over {num_seeds} seeds: {sum(averages['single_pendulum_cip_tts']) / num_seeds}")
    print(f"OL averaged over {num_seeds} seeds: {sum(averages['single_pendulum_ol_tts']) / num_seeds}")

    print()
    print('Cart Pole Time to Stabilization:')
    print(f'Empowerment: {get_cart_pole_tts(cart_pole_empowerment, angle_thresh, dt)}')
    print(f"CIP averaged over {num_seeds} seeds: {sum(averages['cart_pole_cip_tts']) / num_seeds}")
    print(f"OL averaged over {num_seeds} seeds: {sum(averages['cart_pole_ol_tts']) / num_seeds}")

    print()
    print('Double Pendulum Time to Stabilization:')
    print(f'Empowerment: {get_double_pendulum_tts(double_pendulum_empowerment, angle_thresh, dt)}')
    print(f"CIP averaged over {num_seeds} seeds: {sum(averages['double_pendulum_cip_tts']) / num_seeds}")
    print(f"OL averaged over {num_seeds} seeds: {sum(averages['double_pendulum_ol_tts']) / num_seeds}")
