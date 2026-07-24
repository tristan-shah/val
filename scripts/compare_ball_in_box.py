'''
Head-to-head comparison of ball-in-box exploration across three control methods:
  * CIP            (results/BALL_IN_BOX/{component})
  * STATE_ENTROPY  (results/STATE_ENTROPY/BALL_IN_BOX)   -- k-NN particle-entropy planning
  * RANDOM         (results/RANDOM/BALL_IN_BOX)           -- uniform random control

For each method it auto-discovers the seeds present on disk (at a shared restitution `e`),
computes a few exploration statistics per seed, and reports mean +/- std across seeds so
the methods can be compared directly. Optionally writes the table to CSV.

Lives at the top level of scripts/ because it reads runs from several method folders.

Statistics reported (per seed, then aggregated across seeds):
  * pct_states_near_wall   fraction of states inside the outer 0.1 border band of the box
                           (|x| > 0.9 or |y| > 0.9 for the default [-1,1] box)
  * coverage_pct           percent of a (grid x grid) tiling of the box that is visited
  * speed                  mean velocity magnitude
  * contacts_per_episode   number of distinct wall-contact approaches in the trajectory

Example:
    python scripts/compare_ball_in_box.py --restitution 0.5
'''

from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np


CIP_TEMPLATE = (
    'seed={seed}-beta=0.0-e={e}-spawn=True-h=128-shots=256-iter=1-'
    'elite=0.1-smooth=0.1-rho=0.9-dt=0.05-steps=1000'
)
SE_TEMPLATE = (
    'seed={seed}-k={k}-sub=1-beta=0.0-e={e}-spawn=True-h=128-shots=256-iter=1-'
    'elite=0.1-smooth=0.1-rho=0.9-dt=0.05-steps=1000'
)
RANDOM_TEMPLATE = 'seed={seed}-e={e}-spawn=True-dt=0.05-steps=1000'


def discover_seeds(root: Path, template: str):
    '''Seeds (sorted ints) that have a run dir matching `template` (with a {seed} hole) under root.'''
    if not root.exists():
        return []
    regex = re.compile('^' + re.escape(template).replace(re.escape('{seed}'), r'(\d+)') + '$')
    seeds = []
    for d in root.glob(template.format(seed='*')):
        m = regex.match(d.name)
        if m and (d / 'traj.npy').exists():
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def _rising_edges(mask: np.ndarray) -> int:
    '''Number of times `mask` transitions False->True, counting an initial True as one.'''
    if mask.size == 0:
        return 0
    return int(np.sum(mask[1:] & ~mask[:-1]) + int(mask[0]))


def trajectory_stats(X, bounds, wall_thresh, contact_margin, grid):
    '''Compute exploration statistics for a single trajectory X of shape (T+1, 4) = [x, y, vx, vy].'''
    xmin, xmax, ymin, ymax = bounds
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    hx, hy = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)

    p, v = X[:, :2], X[:, 2:]
    ## normalized distance from center along each axis: 0 = center, 1 = wall
    nx = np.abs(p[:, 0] - cx) / hx
    ny = np.abs(p[:, 1] - cy) / hy

    ## fraction of states inside the outer (1 - wall_thresh) border band
    near_wall = (nx > wall_thresh) | (ny > wall_thresh)

    ## a "contact" is a fresh approach into the thin band within contact_margin of a wall
    contact_band = 1.0 - contact_margin
    contacts = _rising_edges(nx > contact_band) + _rising_edges(ny > contact_band)

    ## state coverage: fraction of a grid x grid tiling of the box that is visited
    counts, _, _ = np.histogram2d(
        p[:, 0], p[:, 1], bins=grid, range=[[xmin, xmax], [ymin, ymax]])
    coverage_pct = 100.0 * (counts > 0).mean()

    speed = np.linalg.norm(v, axis=1)

    return {
        'pct_states_near_wall': 100.0 * near_wall.mean(),
        'coverage_pct':         coverage_pct,
        'speed':                float(speed.mean()),
        'contacts_per_episode': float(contacts),
    }


def aggregate(root, template, seeds, bounds, wall_thresh, contact_margin, grid):
    '''Load each seed's trajectory, compute stats, return (mean_dict, std_dict, used_seeds).'''
    rows, used = [], []
    for seed in seeds:
        traj_path = root / template.format(seed=seed) / 'traj.npy'
        if not traj_path.exists():
            print(f'[skip] missing: {traj_path}')
            continue
        X = np.load(traj_path)
        rows.append(trajectory_stats(X, bounds, wall_thresh, contact_margin, grid))
        used.append(seed)
    if not rows:
        return None, None, []
    keys = rows[0].keys()
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    return ({k: float(arr[k].mean()) for k in keys},
            {k: float(arr[k].std()) for k in keys},
            used)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--restitution', type=float, default=0.0,
                        help='Restitution (e) shared by all methods\' runs.')
    parser.add_argument('--component', type=str, default='ol', help='CIP component subdir.')
    parser.add_argument('--k', type=int, default=12, help='State-entropy neighbor count of the runs to load.')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds to use. Default: auto-discover per method.')
    parser.add_argument('--cip_root', type=str, default='results/BALL_IN_BOX')
    parser.add_argument('--se_root', type=str, default='results/STATE_ENTROPY/BALL_IN_BOX')
    parser.add_argument('--random_root', type=str, default='results/RANDOM/BALL_IN_BOX')
    parser.add_argument('--cip_template', type=str, default=CIP_TEMPLATE)
    parser.add_argument('--se_template', type=str, default=SE_TEMPLATE)
    parser.add_argument('--random_template', type=str, default=RANDOM_TEMPLATE)
    parser.add_argument('--bounds', type=float, nargs=4, default=[-1.0, 1.0, -1.0, 1.0],
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    parser.add_argument('--wall_thresh', type=float, default=0.9,
                        help='Normalized distance-from-center above which a coord counts as "near wall".')
    parser.add_argument('--contact_margin', type=float, default=0.05,
                        help='Normalized band next to a wall used to count contact approaches.')
    parser.add_argument('--grid', type=int, default=20, help='Grid resolution for coverage.')
    parser.add_argument('--out', type=str, default=None, help='Optional CSV output path.')
    args = parser.parse_args()

    bounds = tuple(args.bounds)

    ## (label, root, template-with-{seed}-still-open) for each method
    methods = [
        ('CIP',           Path(args.cip_root) / args.component,
         args.cip_template.format(e=args.restitution, seed='{seed}')),
        ('STATE_ENTROPY', Path(args.se_root),
         args.se_template.format(e=args.restitution, k=args.k, seed='{seed}')),
        ('RANDOM',        Path(args.random_root),
         args.random_template.format(e=args.restitution, seed='{seed}')),
    ]

    results = {}
    for label, root, template in methods:
        seeds = args.seeds if args.seeds is not None else discover_seeds(root, template)
        print(f'{label:14s} seeds ({len(seeds)}): {seeds}')
        mean, std, used = aggregate(
            root, template, seeds, bounds, args.wall_thresh, args.contact_margin, args.grid)
        results[label] = (mean, std, used)

    if all(results[m][0] is None for m in results):
        raise SystemExit(f'No runs found for any method at e={args.restitution}. '
                         'Check --restitution / --component / --k / roots.')

    ## ---- print comparison table ----
    stat_order = ['pct_states_near_wall', 'coverage_pct', 'speed', 'contacts_per_episode']
    labels = [m[0] for m in methods]

    def cell(label, key):
        mean, std, _ = results[label]
        if mean is None:
            return '        n/a'
        return f'{mean[key]:8.3f} ± {std[key]:6.3f}'

    print(f'\nBall-in-box exploration comparison  (e={args.restitution}, '
          f'grid={args.grid}x{args.grid}, wall_thresh={args.wall_thresh})')
    print('  ' + '   |   '.join(f'{lab}: {len(results[lab][2])} seeds' for lab in labels) + '\n')

    header = f'{"statistic":22s} | ' + ' | '.join(f'{lab:>17s}' for lab in labels)
    print(header)
    print('-' * len(header))
    for key in stat_order:
        print(f'{key:22s} | ' + ' | '.join(f'{cell(lab, key):>17s}' for lab in labels))

    ## ---- markdown table (copy-paste; GitHub / OpenReview compatible) ----
    def md_cell(label, key):
        mean, std, _ = results[label]
        return 'n/a' if mean is None else f'{mean[key]:.3f} ± {std[key]:.3f}'

    md_headers = ['statistic'] + [f'{lab} (n={len(results[lab][2])})' for lab in labels]
    md_lines = [
        '| ' + ' | '.join(md_headers) + ' |',
        '| ' + ' | '.join(['---'] * len(md_headers)) + ' |',
    ]
    for key in stat_order:
        md_lines.append('| ' + ' | '.join([key] + [md_cell(lab, key) for lab in labels]) + ' |')
    md_table = '\n'.join(md_lines)

    print('\nMarkdown (copy-paste):\n')
    print(md_table)

    ## ---- optional CSV (+ sibling markdown) ----
    if args.out:
        import csv
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            head = ['statistic']
            for lab in labels:
                head += [f'{lab.lower()}_mean', f'{lab.lower()}_std']
            w.writerow(head)
            for key in stat_order:
                row = [key]
                for lab in labels:
                    mean, std, _ = results[lab]
                    row += ['' if mean is None else mean[key], '' if std is None else std[key]]
                w.writerow(row)
        md_path = str(Path(args.out).with_suffix('.md'))
        with open(md_path, 'w') as f:
            f.write(md_table + '\n')
        print(f'\nWrote {args.out} and {md_path}')
