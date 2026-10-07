'''
Figure 5: open-loop information production rate (nats / s) over time during one MPC run per
environment. All four runs use the same planning horizon (512) so the rate is evaluated
consistently across environments; this differs from the per-environment hyperparameters of
Table 3 used for Figure 4. Writes figures/rate.{pdf,png}. Run from the repository root.
'''

from pathlib import Path

from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

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

    dt = 0.01

    cart_pole_ol       = jnp.load('results/CIP/CART_POLE/seed=0-beta=0.0-h=512-shots=1024-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    double_pendulum_ol = jnp.load('results/CIP/DOUBLE_PENDULUM/seed=0-beta=0.0-h=512-shots=512-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    triple_pendulum_ol = jnp.load('results/CIP/TRIPLE_PENDULUM/seed=0-gear=25.0-beta=3.0-h=512-shots=2048-iter=3-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]
    humulum_ol         = jnp.load('results/CIP/HUMULUM/seed=1-beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200/hist.npy')[:, 1]

    T = humulum_ol.shape[0]
    t = jnp.linspace(0.0, T * dt, T)

    # Determine y-limits from data to set the break automatically
    low_curves = [cart_pole_ol, double_pendulum_ol, triple_pendulum_ol]
    low_min  = float(min(c.min() for c in low_curves))
    low_max  = float(max(c.max() for c in low_curves))
    high_min = float(humulum_ol.min())
    high_max = float(humulum_ol.max())

    low_pad  = max((low_max  - low_min)  * 0.12, 0.5)
    high_pad = max((high_max - high_min) * 0.12, 0.5)

    bot_ylim = (low_min  - low_pad,  low_max  + low_pad)
    top_ylim = (high_min - high_pad, high_max + high_pad)

    # Scale panel heights proportionally to their data ranges, clamped so the
    # top panel is at least 35% as tall as the bottom panel.
    top_ratio = max((top_ylim[1] - top_ylim[0]) / (bot_ylim[1] - bot_ylim[0]), 0.35)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(3.0, 2.2),
        gridspec_kw={'height_ratios': [top_ratio, 1.0], 'hspace': 0.06},
    )

    colors = ['C0', 'C1', 'C2', 'C3']
    labels = ['Cart pole', 'Double pendulum', 'Triple pendulum', 'Gibbon']
    curves = [cart_pole_ol, double_pendulum_ol, triple_pendulum_ol, humulum_ol]

    for ax in (ax_top, ax_bot):
        for curve, label, color in zip(curves, labels, colors):
            ax.plot(t, curve, label=label, color=color, zorder=3)

    ax_top.set_ylim(*top_ylim)
    ax_bot.set_ylim(*bot_ylim)

    # Remove the spines that face the break
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_bot.set_xlim(0.0, 12.0)
    ax_bot.set_xticks([0, 6, 12])
    ax_bot.set_xlabel('Time (s)')

    # Integer ticks, pruned away from the break to avoid crowding
    ax_top.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=True, prune='lower'))
    ax_bot.yaxis.set_major_locator(ticker.MaxNLocator(4, integer=True, prune='upper'))

    fig.supylabel('nats/s', fontsize=7, x=0.02)

    fig.tight_layout(pad=0.4)

    # Draw break marks in figure coordinates (after layout) so they are
    # identical in physical size regardless of the panel height ratio.
    pos_top = ax_top.get_position()
    pos_bot = ax_bot.get_position()
    dx, dy = 0.010, 0.007   # half-extents in figure-fraction units
    kw = dict(transform=fig.transFigure, color='k', linewidth=0.8, clip_on=False)
    for x_fig in [pos_top.x0, pos_top.x1]:
        for y_fig in [pos_top.y0, pos_bot.y1]:
            fig.add_artist(Line2D([x_fig - dx, x_fig + dx],
                                  [y_fig - dy, y_fig + dy], **kw))

    # Single legend — pull from one axis only to avoid duplicates.
    # Reorder to [0, 2, 1, 3] so a 2-column legend reads:
    #   Cart pole        Triple pendulum
    #   Double pendulum  Gibbon
    handles, labels = ax_bot.get_legend_handles_labels()
    handles = [handles[i] for i in [0, 2, 1, 3]]
    labels  = [labels[i]  for i in [0, 2, 1, 3]]
    fig.legend(handles, labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=6)
    Path('figures').mkdir(exist_ok=True)
    fig.savefig('figures/rate.pdf', bbox_inches='tight')
    fig.savefig('figures/rate.png', dpi=600, bbox_inches='tight')
    print('Saved figures/rate.pdf and figures/rate.png')
