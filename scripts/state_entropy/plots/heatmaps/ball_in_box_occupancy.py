'''
State-occupancy heatmap for the ball-in-box STATE-ENTROPY (APT) experiment.

Aggregates the (px, py) positions from every seed's saved `traj.npy` into a single
2D histogram over the box and plots the occupancy distribution (fraction of time the
ball spends in each cell), pooled across all seeds.

Add more seeds by extending --seeds; runs that don't exist yet are skipped with a
warning, so it's safe to point this at a partially-completed sweep.

Example:
    python scripts/state_entropy/plots/heatmaps/ball_in_box_occupancy.py --restitution 0.5
'''

from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


def discover_seeds(root: Path, template: str):
    '''
    Find every seed that has a run directory matching `template` under `root`.
    Returns a sorted list of the integer seeds present on disk.
    '''
    ## glob for any seed, then pull the integer back out of each matching dir name
    pattern = template.format(seed='*')
    regex = re.compile('^' + re.escape(template).replace(re.escape('{seed}'), r'(\d+)') + '$')
    seeds = []
    for d in root.glob(pattern):
        m = regex.match(d.name)
        if m and (d / 'traj.npy').exists():
            seeds.append(int(m.group(1)))
    return sorted(seeds)

## run-directory name template; {seed}, {e} (restitution) and {k} (neighbors) are
## substituted in. Change this (or pass --template) if other hyperparameters change.
DEFAULT_TEMPLATE = (
    'seed={seed}-k={k}-sub=1-beta=0.0-e={e}-spawn=True-h=128-shots=256-iter=1-'
    'elite=0.1-smooth=0.1-rho=0.9-dt=0.05-steps=1000'
)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to pool. Default: auto-discover every seed present '
                             'on disk for this path type.')
    parser.add_argument('--root', type=str, default='results/STATE_ENTROPY/BALL_IN_BOX',
                        help='Results root; runs live in {root}/{template}')
    parser.add_argument('--template', type=str, default=DEFAULT_TEMPLATE,
                        help='Run-dir name template containing {seed}, {e} and {k} placeholders')
    parser.add_argument('--restitution', type=float, default=0.0,
                        help='Restitution (e) of the runs to load; fills {e} in the template.')
    parser.add_argument('--k', type=int, default=12,
                        help='APT neighbor count (k) of the runs to load; fills {k} in the template.')
    parser.add_argument('--bounds', type=float, nargs=4, default=[-1.0, 1.0, -1.0, 1.0],
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help='Box extent used for the histogram range')
    parser.add_argument('--bins', type=int, default=100, help='Histogram bins per axis')
    parser.add_argument('--cmap', type=str, default='magma')
    parser.add_argument('--out', type=str, default=None,
                        help='Output stem (without extension); defaults next to the runs')
    args = parser.parse_args()

    xmin, xmax, ymin, ymax = args.bounds
    root = Path(args.root)

    ## fill restitution and k into the template, keeping {seed} for per-seed substitution
    template = args.template.format(e=args.restitution, k=args.k, seed='{seed}')

    ## default: pool every seed available on disk for this path type
    seeds = args.seeds
    if seeds is None:
        seeds = discover_seeds(root, template)
        if not seeds:
            raise SystemExit(
                f'No seeds found under {root} matching template. '
                'Check --root / --restitution / --k / --template.')
        print(f'Auto-discovered seeds: {seeds}')

    ## collect positions from every available seed
    positions = []
    used_seeds = []
    for seed in seeds:
        traj_path = root / template.format(seed=seed) / 'traj.npy'
        if not traj_path.exists():
            print(f'[skip] missing: {traj_path}')
            continue
        X = np.load(traj_path)          # (steps + 1, 4) -> [px, py, vx, vy]
        positions.append(X[:, :2])
        used_seeds.append(seed)

    if not positions:
        raise SystemExit('No trajectories found. Check --root / --restitution / --k / --template.')

    positions = np.concatenate(positions, axis=0)
    print(f'Pooled {positions.shape[0]} states from seeds {used_seeds}')

    ## 2D histogram of position occupancy, normalized to a probability distribution
    counts, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=args.bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    occupancy = counts / counts.sum()

    ## imshow expects (row=y, col=x)
    heatmap = occupancy.T

    stem = args.out or str(
        root / f'occupancy-state_entropy-e={args.restitution}-k={args.k}-nseeds={len(used_seeds)}-bins={args.bins}'
    )
    np.save(f'{stem}.npy', heatmap)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap,
        origin='lower',
        extent=[xmin, xmax, ymin, ymax],
        aspect='equal',
        cmap=args.cmap,
    )
    ## box outline
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color='white', linewidth=1, alpha=0.6)
    fig.colorbar(im, ax=ax, label='Occupancy (fraction of time)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Ball-in-Box state occupancy (state entropy, {len(used_seeds)} seeds)')
    fig.tight_layout()
    fig.savefig(f'{stem}.png', dpi=300)
    print(f'Saved {stem}.png and {stem}.npy')
    plt.show()
