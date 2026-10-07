'''
Figure 6 (Appendix A.1): ball-in-box occupancy, three panels in a row.

    left    schematic of the box with the ball and a short fading trail (drawn from a
            real CIP trajectory) to show that the ball is moving around
    middle  position-occupancy heatmap for CIP, pooled over N_SEEDS seeds
    right   position-occupancy heatmap for uniform random control, pooled over N_SEEDS seeds

Both heatmaps share a single colour scale so they are directly comparable (25 bins,
restitution 0.5). Writes figures/ball_in_box.{pdf,png}. Run from the repository root.

    python scripts/cip/plots/plot_ball_in_box.py
'''

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESTITUTION = 0.5
N_SEEDS     = 50
SEEDS       = list(range(N_SEEDS))
BINS        = 25
BOUNDS      = (-1.0, 1.0, -1.0, 1.0)     # (xmin, xmax, ymin, ymax)
CMAP        = 'magma'

CIP_ROOT     = Path('results/CIP/BALL_IN_BOX')
CIP_TEMPLATE = (
    'seed={seed}-beta=0.0-e={e}-spawn=True-h=128-shots=256-iter=1-'
    'elite=0.1-smooth=0.1-rho=0.9-dt=0.05-steps=1000'
)
RANDOM_ROOT     = Path('results/RANDOM/BALL_IN_BOX')
RANDOM_TEMPLATE = 'seed={seed}-e={e}-spawn=True-dt=0.05-steps=1000'

## schematic (left panel): which CIP trajectory to draw the trail from, and which
## window of it. The ball is drawn at the last step of the window.
SCHEMATIC_SEED  = 0
SCHEMATIC_STEP  = 66
SCHEMATIC_TRAIL = 66
BALL_RADIUS     = 0.07


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_positions(root, template, seeds):
    '''Pool (px, py) from every seed's traj.npy; missing runs are skipped with a warning.'''
    positions, used = [], []
    for seed in seeds:
        p = root / template.format(seed=seed, e=RESTITUTION) / 'traj.npy'
        if not p.exists():
            print(f'  Missing: {p}')
            continue
        X = np.load(p)                      # (steps + 1, 4) -> [px, py, vx, vy]
        positions.append(X[:, :2])
        used.append(seed)
    if not positions:
        raise FileNotFoundError(f'No trajectories found under {root}')
    print(f'  Pooled {len(used)} seeds from {root}')
    return np.concatenate(positions, axis=0)


def occupancy(positions):
    '''2D position histogram over the box, normalised to a distribution. Returns (row=y, col=x).'''
    xmin, xmax, ymin, ymax = BOUNDS
    counts, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=BINS, range=[[xmin, xmax], [ymin, ymax]])
    return (counts / counts.sum()).T


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def draw_box(ax, color='black', linewidth=1.0):
    xmin, xmax, ymin, ymax = BOUNDS
    ax.plot([xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color=color, linewidth=linewidth, solid_capstyle='projecting',
            clip_on=False, zorder=4)


def draw_schematic(ax, X):
    '''Box + ball + fading trail, no axes.'''
    xmin, xmax, ymin, ymax = BOUNDS
    lo = max(0, SCHEMATIC_STEP - SCHEMATIC_TRAIL)
    seg = X[lo:SCHEMATIC_STEP + 1, :2]

    ## trail: one line segment per step, alpha and width both ramping up toward the
    ## ball (comet-style). Butt caps so overlapping translucent segments don't bead.
    pts   = seg.reshape(-1, 1, 2)
    lines = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n     = len(lines)
    alpha = np.linspace(0.05, 0.8, n)
    width = np.linspace(0.3, 1.8, n)
    lc = LineCollection(lines, colors=[(0.0, 0.0, 0.0, a) for a in alpha],
                        linewidths=width, capstyle='butt', joinstyle='round', zorder=2)
    ax.add_collection(lc)

    ## ball
    ax.add_patch(Circle(seg[-1], BALL_RADIUS, facecolor='C3', edgecolor='none', zorder=3))

    ## box outline fills the axes exactly, so it matches the heatmap panels' extent
    draw_box(ax, linewidth=1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_heatmap(ax, heatmap, vmax):
    xmin, xmax, ymin, ymax = BOUNDS
    im = ax.imshow(heatmap, origin='lower', extent=[xmin, xmax, ymin, ymax],
                   aspect='equal', cmap=CMAP, vmin=0.0, vmax=vmax,
                   interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    return im


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    plt.rcParams.update({
        'font.size':        7,
        'axes.titlesize':   8,
        'axes.labelsize':   7,
        'xtick.labelsize':  6,
        'ytick.labelsize':  6,
        'legend.fontsize':  6,
        'lines.linewidth':  1.0,
    })

    print('Loading CIP ...')
    cip_positions = load_positions(CIP_ROOT, CIP_TEMPLATE, SEEDS)
    print('Loading random ...')
    rnd_positions = load_positions(RANDOM_ROOT, RANDOM_TEMPLATE, SEEDS)

    cip_heat = occupancy(cip_positions)
    rnd_heat = occupancy(rnd_positions)
    vmax = max(cip_heat.max(), rnd_heat.max())

    schematic_X = np.load(
        CIP_ROOT / CIP_TEMPLATE.format(seed=SCHEMATIC_SEED, e=RESTITUTION) / 'traj.npy')

    fig, axes = plt.subplots(1, 3, figsize=(5.2, 1.9))
    ax_box, ax_cip, ax_rnd = axes

    draw_schematic(ax_box, schematic_X)
    draw_heatmap(ax_cip, cip_heat, vmax)
    im = draw_heatmap(ax_rnd, rnd_heat, vmax)

    ax_box.set_title('Ball in box')
    ax_cip.set_title('CIP (ours)')
    ax_rnd.set_title('Random')

    fig.tight_layout(pad=0.4, w_pad=1.0)

    ## single shared colourbar to the right of the two heatmaps
    pos = ax_rnd.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.015, pos.y1 - pos.y0])
    ## no numeric ticks: the shared scale is what matters, not the absolute values
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label('Occupancy density', labelpad=4)
    cb.set_ticks([])
    cb.outline.set_linewidth(0.6)

    Path('figures').mkdir(exist_ok=True)
    fig.savefig('figures/ball_in_box.pdf', bbox_inches='tight')
    fig.savefig('figures/ball_in_box.png', dpi=600, bbox_inches='tight')
    print('Saved figures/ball_in_box.pdf and figures/ball_in_box.png')
