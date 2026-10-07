'''
Reproduces Figure 3 of the paper: KSE estimation on the Lorenz system in two chaotic parameter
regimes, comparing the continuous-time Riccati estimator ("Ours") against the discrete QR baseline
(Benettin et al.), with the known value as a reference line.

    A  "lorenz_standard"     sigma = 10, rho = 28,    beta = 8/3   known KSE ~ 0.906 (Sprott 1997)
    B  "lorenz_nonstandard"  sigma = 16, rho = 45.92, beta = 4.0   known KSE ~ 1.498 (Wolf et al. 1985: 2.16 bits)

Both estimators run on the RK4 step of the Lorenz flow at dt = 0.005 in double precision from
x0 = (0.01, 0.01, 0.01) for t1 = 1000, with no burn-in so the convergence transient is visible.

    python scripts/kse_estimation/plot_lorenz_kse.py                 # both sets, then the figure
    python scripts/kse_estimation/plot_lorenz_kse.py --set A         # one set only
    python scripts/kse_estimation/plot_lorenz_kse.py --plot-only     # rebuild the figure from cached .npy

Outputs go to <out>/lorenz_standard/ and <out>/lorenz_nonstandard/ as ours.npy, qr.npy and ts.npy,
and the figure to <out>/lorenz_kse.pdf and <out>/lorenz_kse.png. Existing files are not overwritten
unless --force is given.
'''

import argparse
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from val.kse import build_attractor, make_evaluate
from val.kse.estimators import make_continuous_riccati, make_discrete_qr


PARAM_SETS = {
    'A': dict(
        dir = 'lorenz_standard',
        sigma = 10.0, rho = 28.0, beta = 8 / 3,
        known_kse = 0.906,                      ## https://sprott.physics.wisc.edu/chaos/lorenzle.htm
        ylim = (0.2, 1.25),
        title = 'Parameter Set A'),
    'B': dict(
        dir = 'lorenz_nonstandard',
        sigma = 16.0, rho = 45.92, beta = 4.0,
        known_kse = float(np.log(2) * 2.16),    ## Wolf et al. 1984 report 2.16 bits
        ylim = (0.5, 2.0),
        title = 'Parameter Set B'),
}

T1 = 1000.0
DT = 0.005
X0 = jnp.array([0.01, 0.01, 0.01])

DEFAULT_OUT = Path(__file__).resolve().parents[2] / 'results' / 'KSE'


def check_no_overwrite(targets, force: bool) -> None:

    existing = [p for p in targets if p.exists()]
    if existing and not force:
        raise SystemExit(
            'Refusing to overwrite existing files:\n  '
            + '\n  '.join(str(p) for p in existing)
            + '\nPass --force to overwrite, or choose another --out directory.')


def run_set(key: str, out_root: Path, force: bool) -> None:

    cfg = PARAM_SETS[key]
    out_dir = out_root / cfg['dir']
    out_dir.mkdir(parents = True, exist_ok = True)
    check_no_overwrite([out_dir / n for n in ('ours.npy', 'qr.npy', 'ts.npy')], force)

    print(f'[set {key}] {cfg["dir"]}: sigma={cfg["sigma"]}, rho={cfg["rho"]}, '
          f'beta={cfg["beta"]:.4f}, t1={T1}, dt={DT}', flush = True)

    ## no burn-in: the figure shows convergence from the initial condition
    attractor = build_attractor('lorenz', dt = DT, burn_in = 0,
                                sigma = cfg['sigma'], rho = cfg['rho'], beta = cfg['beta'])
    steps = int(round(T1 / DT))
    ts = np.arange(1, steps + 1) * DT

    ours = jax.jit(make_evaluate(attractor, make_continuous_riccati(attractor), steps))(X0)['kse']
    qr = jax.jit(make_evaluate(attractor, make_discrete_qr(attractor), steps))(X0)['kse']

    np.save(out_dir / 'ours.npy', np.asarray(ours))
    np.save(out_dir / 'qr.npy', np.asarray(qr))
    np.save(out_dir / 'ts.npy', ts)

    print(f'[set {key}] final KSE  ours={float(ours[-1]):.4f}  qr={float(qr[-1]):.4f}  '
          f'known={cfg["known_kse"]:.4f}')
    print(f'[set {key}] wrote {out_dir}/{{ours,qr,ts}}.npy', flush = True)


def make_plot(out_root: Path, force: bool) -> None:

    check_no_overwrite([out_root / 'lorenz_kse.pdf', out_root / 'lorenz_kse.png'], force)

    plt.rcParams.update({
        'font.size':        7,
        'axes.titlesize':   8,
        'axes.labelsize':   7,
        'xtick.labelsize':  6,
        'ytick.labelsize':  6,
        'legend.fontsize':  6,
        'lines.linewidth':  1.0,
    })

    fig, axes = plt.subplots(1, 2, figsize = (7.0, 1.8), sharey = False)

    for ax, key in zip(axes, ('A', 'B')):
        cfg = PARAM_SETS[key]
        ours = np.load(out_root / cfg['dir'] / 'ours.npy')
        qr = np.load(out_root / cfg['dir'] / 'qr.npy')
        ts = np.load(out_root / cfg['dir'] / 'ts.npy')

        ax.axhline(cfg['known_kse'], color = 'red', linestyle = 'dashed',
                   linewidth = 0.8, label = 'Known', zorder = 3)
        ax.plot(ts, qr, label = 'QR', zorder = 1, color = 'blue')
        ax.plot(ts, ours, label = 'Ours', zorder = 2, color = 'green')

        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(*cfg['ylim'])
        ax.set_title(cfg['title'])
        ax.set_xlabel('Time (s)')

    axes[0].set_ylabel('nats / s')

    ## shared legend below both panels, with Ours first
    handles, labels = axes[0].get_legend_handles_labels()
    order = ['Ours', 'QR', 'Known']
    handles = [handles[labels.index(l)] for l in order]
    fig.legend(handles, order, loc = 'lower center', ncol = 3,
               bbox_to_anchor = (0.5, -0.08), frameon = False, fontsize = 7)

    fig.tight_layout(pad = 0.5, w_pad = 1.0)
    fig.savefig(out_root / 'lorenz_kse.pdf', bbox_inches = 'tight')
    fig.savefig(out_root / 'lorenz_kse.png', dpi = 600, bbox_inches = 'tight')
    plt.close(fig)

    print(f'[plot] wrote {out_root}/lorenz_kse.pdf and lorenz_kse.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', choices = ['A', 'B', 'both'], default = 'both',
                        help = 'which parameter set to run (default: both)')
    parser.add_argument('--out', type = Path, default = DEFAULT_OUT,
                        help = f'output root directory (default: {DEFAULT_OUT})')
    parser.add_argument('--force', action = 'store_true', help = 'overwrite existing output files')
    parser.add_argument('--plot-only', action = 'store_true',
                        help = 'skip the simulations and only rebuild the figure from the .npy files in --out')
    args = parser.parse_args()

    if not args.plot_only:
        for key in (['A', 'B'] if args.set == 'both' else [args.set]):
            run_set(key, args.out, args.force)

    needed = [args.out / PARAM_SETS[k]['dir'] / n for k in ('A', 'B') for n in ('ours.npy', 'qr.npy', 'ts.npy')]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print('\n[plot] skipped: the two-panel figure needs both sets. Missing:\n  '
              + '\n  '.join(str(p) for p in missing))
    else:
        make_plot(args.out, args.force)
