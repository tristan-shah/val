import jax
from pathlib import Path
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step, make_unroll

def load_average_history(base_path, method, num_seeds, horizon, dt):
    '''Load history arrays for multiple seeds and return stacked array.'''
    histories = []
    for seed in range(num_seeds):
        path = Path(f'{base_path}/{method}/seed={seed}-h={horizon}-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt={dt}')
        hist = jnp.load(path / 'hist.npy')
        histories.append(hist)
    return jnp.stack(histories).mean(axis = 0)  # shape: (num_seeds, time_steps, features) <- average over seeds


def tv_controllability_gramian_stable(As: jnp.ndarray, Bs: jnp.ndarray) -> jnp.ndarray:
    """
    Compute finite-horizon controllability Gramian along a trajectory in a numerically stable way.

    Args:
        As: (T, n, n) - A_t matrices along trajectory
        Bs: (T, n, m) - B_t matrices along trajectory

    Returns:
        W: (n, n) - PSD Gramian at final time
    """
    # Force float64 for higher precision
    As = As.astype(jnp.float64)
    Bs = Bs.astype(jnp.float64)

    n = As.shape[1]
    W0 = jnp.zeros((n, n), dtype=jnp.float64)

    def step(W, inputs):
        A, B = inputs
        W_next = A @ W @ A.T + B @ B.T
        # Symmetrize to ensure PSD numerically
        W_next = 0.5 * (W_next + W_next.T)
        return W_next, W_next

    WN, W_hist = jax.lax.scan(step, W0, (As, Bs))
    return WN


# def tv_controllability_gramian(As: jnp.ndarray, Bs: jnp.ndarray) -> jnp.ndarray:
#     """
#     Compute finite-horizon controllability Gramian along a trajectory.

#     As: (T, n, n) - A_t matrices along trajectory
#     Bs: (T, n, m) - B_t matrices along trajectory
#     Returns:
#         W: (n, n) - Gramian at final time
#     """
#     n = As.shape[1]
#     W0 = jnp.zeros((n, n), dtype=As.dtype)

#     def step(W, inputs):
#         A, B = inputs
#         W_next = A @ W @ A.T + B @ B.T
#         return W_next, W_next

#     WN, W_hist = jax.lax.scan(step, W0, (As, Bs))
#     return WN

if __name__ == '__main__':

    num_seeds = 6
    ## task cartpole
    root = 'results/CIP/CART_POLE/exponential_domain'
    horizon = 400
    dt = 0.01

    cip_hist = load_average_history(root, 'cip', num_seeds, horizon, dt)
    ol_hist = load_average_history(root, 'ol', num_seeds, horizon, dt)

    print(cip_hist.mean(axis = 0))
    print(ol_hist.mean(axis = 0))







    # dyn = Dynamics('xml/pendulum.xml', dt = 0.05)
    dyn = Dynamics('xml/cart_pole.xml', dt = 0.01)
    step = make_step(dyn)
    traj_linerize = jax.vmap(jax.jacfwd(step, argnums = (0, 1)))

    cip_log_det = []
    ol_log_det = []

    for seed in range(num_seeds):
        # seed = 0
        # cip_root = Path(f'/Users/tristanshah/Desktop/code/val/results/CIP/SINGLE_PENDULUM/exponential_domain/cip/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05')
        # ol_root = Path(f'/Users/tristanshah/Desktop/code/val/results/CIP/SINGLE_PENDULUM/exponential_domain/ol/seed={seed}-h=150-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.05')

        cip_root = Path(f'/Users/tristanshah/Desktop/code/val/results/CIP/CART_POLE/exponential_domain/cip/seed={seed}-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01')
        ol_root = Path(f'/Users/tristanshah/Desktop/code/val/results/CIP/CART_POLE/exponential_domain/ol/seed={seed}-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01')
        
        
        X = jnp.load(cip_root / 'traj.npy')
        U = jnp.load(cip_root / 'U.npy')
        fx, fu = traj_linerize(X[:-1], U)
        cip_W = tv_controllability_gramian_stable(fx, fu)
        cip_vol = jnp.linalg.slogdet(cip_W).logabsdet
        print(f'CIP Log Det: {cip_vol}')


        X = jnp.load(ol_root / 'traj.npy')
        U = jnp.load(ol_root / 'U.npy')
        fx, fu = traj_linerize(X[:-1], U)
        ol_W = tv_controllability_gramian_stable(fx, fu)
        ol_vol = jnp.linalg.slogdet(ol_W).logabsdet
        print(f'OL Log Det: {ol_vol}')

        cip_log_det.append(cip_vol)
        ol_log_det.append(ol_vol)

    cip_log_det = jnp.array(cip_log_det)
    ol_log_det = jnp.array(ol_log_det)

    print(f'CIP controllability grammian logdet: {cip_log_det.mean()} +- {cip_log_det.std()}')
    print(f'OL controllability grammian logdet: {ol_log_det.mean()} +- {ol_log_det.std()}')