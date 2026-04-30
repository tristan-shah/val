# import os
# os.environ['MUJOCO_GL'] = 'egl'

from argparse import ArgumentParser

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from val import Dynamics, make_step
from val.cip import make_compute_cip

'''
## interpretation of the state components
xt[0] = pos
xt[1] = angle
xt[2] = translational vel
xt[3] = angular vel
'''

MODES = {
    'angle_angvel': dict(
        xi=1, xlo=0.0,       xhi=2 * np.pi, xlabel='Angle (rad)',
        yi=3, ylo=-8.0,      yhi=8.0,        ylabel='Angular velocity (rad/s)',
        nx_arg='n_angle', ny_arg='n_angvel',
    ),
    'pos_angle': dict(
        xi=0, xlo=-2.4,      xhi=2.4,        xlabel='Cart position (m)',
        yi=1, ylo=0.0,       yhi=2 * np.pi,  ylabel='Angle (rad)',
        nx_arg='n_pos', ny_arg='n_angle',
    ),
}

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode', choices=list(MODES), default='angle_angvel',
                        help='Which pair of state dimensions to sweep')
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--n1', type=int, default=200, help='Grid size along x-axis')
    parser.add_argument('--n2', type=int, default=200, help='Grid size along y-axis')
    args = parser.parse_args()

    dt = 0.01
    horizon = args.horizon
    mode = MODES[args.mode]

    ## initialize dynamics
    dyn = Dynamics('xml/cart_pole.xml', dt=dt)

    compute_cip = make_compute_cip(dyn, component='ol')

    ## vmap over states (first arg), broadcast a single U across the batch
    batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes=(0, None)))

    x_vals = jnp.linspace(mode['xlo'], mode['xhi'], args.n1)
    y_vals = jnp.linspace(mode['ylo'], mode['yhi'], args.n2)
    xs, ys = jnp.meshgrid(x_vals, y_vals)   # (n2, n1)
    n = args.n1 * args.n2

    states = jnp.zeros((n, dyn.state_dim))
    states = states.at[:, mode['xi']].set(xs.ravel())
    states = states.at[:, mode['yi']].set(ys.ravel())

    U = jnp.zeros((horizon, dyn.control_dim))

    print(f'Computing open-loop entropy [{args.mode}, {args.n1}x{args.n2}]...')
    values, _ = batch_compute_cip(states, U)
    heatmap = np.array(values).reshape(args.n2, args.n1)

    stem = f'h={horizon}-{args.mode}-cart_pole-heatmap'
    jnp.save(f'{stem}.npy', heatmap)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap,
        origin='lower',
        extent=[mode['xlo'], mode['xhi'], mode['ylo'], mode['yhi']],
        aspect='auto',
        cmap='viridis',
    )
    fig.colorbar(im, ax=ax, label='Entropy Production Rate (nats/s)')
    ax.set_xlabel(mode['xlabel'])
    ax.set_ylabel(mode['ylabel'])
    ax.set_title(f'Cart Pole')
    fig.tight_layout()
    fig.savefig(f'{stem}.png', dpi=300)
    plt.show()
