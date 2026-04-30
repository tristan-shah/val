# import os
# os.environ['MUJOCO_GL'] = 'egl'

from argparse import ArgumentParser
from pathlib import Path

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from val import Dynamics
from val.cip import make_compute_cip

'''
## interpretation of the state components (triple pendulum, state_dim=6)
xt[0] = angle1        (hinge1, absolute)
xt[1] = angle2        (hinge2, relative to link 1)
xt[2] = angle3        (hinge3, relative to link 2)
xt[3] = angular velocity 1
xt[4] = angular velocity 2
xt[5] = angular velocity 3
'''

RESULTS  = Path('results/CIP/TRIPLE_PENDULUM/ol')
SUFFIX   = 'gear=25.0-beta=2.5-h=128-shots=2048-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=2400'
SEEDS    = list(range(10))
GEAR     = 25.0

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--horizon',    type=int,   default=64)
    parser.add_argument('--n_bins',     type=int,   default=100,  help='Heatmap grid resolution')
    parser.add_argument('--batch_size', type=int,   default=2048, help='States per entropy batch')
    parser.add_argument('--reduction',  choices=['pca', 'umap'], default='pca')
    parser.add_argument('--pos_only',   action='store_true', help='Zero velocities before reduction (embed angles only)')
    args = parser.parse_args()

    dt      = 0.01
    horizon = args.horizon

    ## initialize dynamics with the same gear override used during the runs
    dyn = Dynamics('xml/triple_pendulum.xml', dt=dt)
    dyn.mjx_model = dyn.mjx_model.replace(
        actuator_gear=dyn.mjx_model.actuator_gear.at[:, 0].set(GEAR)
    )

    compute_cip       = make_compute_cip(dyn, component='ol')
    batch_compute_cip = jax.jit(jax.vmap(compute_cip, in_axes=(0, None)))

    ## load all trajectory states
    all_states = []
    for seed in SEEDS:
        traj_path = RESULTS / f'retest-seed={seed}-{SUFFIX}' / 'traj.npy'
        if traj_path.exists():
            all_states.append(np.array(jnp.load(traj_path)))
        else:
            print(f'Missing: {traj_path}')

    if not all_states:
        raise FileNotFoundError('No trajectories found.')

    all_states = np.concatenate(all_states, axis=0)   # (N, state_dim)
    print(f'Loaded {all_states.shape[0]} states from {len(SEEDS)} seeds')

    ## compute entropy in chunks to avoid OOM
    U         = jnp.zeros((horizon, dyn.control_dim))
    n_states  = all_states.shape[0]
    entropies = np.empty(n_states)

    print(f'Computing open-loop entropy (horizon={horizon}, batch={args.batch_size})...')
    for start in range(0, n_states, args.batch_size):
        end    = min(start + args.batch_size, n_states)
        chunk  = jnp.array(all_states[start:end])
        vals, _ = batch_compute_cip(chunk, U)
        entropies[start:end] = np.array(vals)
        print(f'  {end}/{n_states}')

    ## encode angles as (cos, sin) so periodicity is respected by the reducer
    ## state layout: [angle1, angle2, angle3, vel1, vel2, vel3]
    angles = all_states[:, :3]
    vels   = all_states[:, 3:] if not args.pos_only else np.zeros_like(all_states[:, 3:])
    features = np.concatenate([np.cos(angles), np.sin(angles), vels], axis=1)

    ## dimensionality reduction
    print(f'Fitting {args.reduction.upper()}...')
    if args.reduction == 'pca':
        reducer = PCA(n_components=2)
        coords  = reducer.fit_transform(features)
        var     = reducer.explained_variance_ratio_
        xlabel  = f'PC1 ({var[0]*100:.1f}% var)'
        ylabel  = f'PC2 ({var[1]*100:.1f}% var)'
    else:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=0)
        coords  = reducer.fit_transform(features)
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'

    ## bin states into 2D grid and average entropy per cell
    x, y   = coords[:, 0], coords[:, 1]
    n_bins = args.n_bins
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=n_bins, weights=entropies)
    counts,  _,      _      = np.histogram2d(x, y, bins=n_bins)
    with np.errstate(invalid='ignore'):
        heatmap = np.where(counts > 0, heatmap / counts, np.nan)

    stem = f'h={horizon}-{args.reduction}-triple_pendulum-heatmap'
    np.save(f'{stem}.npy', heatmap)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap.T,
        origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='auto',
        cmap='viridis',
    )
    fig.colorbar(im, ax=ax, label='Entropy Production Rate (nats/s)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Triple Pendulum')
    fig.tight_layout()
    fig.savefig(f'{stem}.png', dpi=300)
    plt.show()
