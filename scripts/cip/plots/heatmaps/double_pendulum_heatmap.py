# import os
# os.environ['MUJOCO_GL'] = 'egl'

from argparse import ArgumentParser

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from val import Dynamics
from val.cip import make_compute_cip

'''
## interpretation of the state components
xt[0] = angle1        (hinge1, absolute world-frame angle of link 1)
xt[1] = angle2        (hinge2, angle of link 2 RELATIVE to link 1)
xt[2] = angular velocity 1
xt[3] = angular velocity 2

Absolute angle of link 2 = angle1 + angle2
Fully extended upright: abs_angle1=π, abs_angle2=π  →  angle1=π, angle2=0
'''

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--n1', type=int, default=200, help='Grid size along angle1 axis')
    parser.add_argument('--n2', type=int, default=200, help='Grid size along angle2 axis')
    args = parser.parse_args()

    dt = 0.01
    horizon = args.horizon

    ## initialize dynamics
    dyn = Dynamics('xml/double_pendulum.xml', dt=dt, integrator='implicitfast')

    compute_cip = make_compute_cip(dyn, component='ol')

    ## vmap over states (first arg), broadcast a single U across the batch
    batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes=(0, None)))

    ## grid over absolute world-frame angles; both centered on π (fully upright)
    abs_angle1_vals = jnp.linspace(0.0, 2 * jnp.pi, args.n1)
    abs_angle2_vals = jnp.linspace(0.0, 2 * jnp.pi, args.n2)
    abs_a1, abs_a2 = jnp.meshgrid(abs_angle1_vals, abs_angle2_vals)   # (n2, n1)
    n = args.n1 * args.n2

    ## convert to MuJoCo relative angles: rel_angle2 = abs_angle2 - abs_angle1
    rel_a2 = abs_a2 - abs_a1

    ## velocities fixed at zero
    states = jnp.zeros((n, dyn.state_dim))
    states = states.at[:, 0].set(abs_a1.ravel())
    states = states.at[:, 1].set(rel_a2.ravel())

    U = jnp.zeros((horizon, dyn.control_dim))

    print(f'Computing open-loop entropy [abs angle1 vs abs angle2, {args.n1}x{args.n2}]...')
    values, _ = batch_compute_cip(states, U)
    heatmap = np.array(values).reshape(args.n2, args.n1)

    stem = f'h={horizon}-double_pendulum-heatmap'
    jnp.save(f'{stem}.npy', heatmap)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap,
        origin='lower',
        extent=[0.0, 2 * np.pi, 0.0, 2 * np.pi],
        aspect='auto',
        cmap='viridis',
    )
    fig.colorbar(im, ax=ax, label='Entropy Production Rate (nats/s)')
    ax.set_xlabel('Absolute angle 1 (rad)')
    ax.set_ylabel('Absolute angle 2 (rad)')
    ax.set_title('Double Pendulum')
    fig.tight_layout()
    fig.savefig(f'{stem}.png', dpi=300)
    plt.show()
