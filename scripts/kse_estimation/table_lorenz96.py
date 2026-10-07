'''
Reproduces Table 2 of the paper: the sum of positive Lyapunov exponents of Lorenz-96 at
N = 10, 25, 50, 75, 100, estimated with the discrete Riccati recursion of Equation (10) ("Ours")
and with discrete QR. Both methods run on the same trajectories (same initial conditions) for
time T at dt = 0.01 in double precision, after 500 burn-in steps for the state and another 500
for the estimator; the final estimate is averaged over several initial conditions.

    python scripts/kse_estimation/table_lorenz96.py                     # paper settings: T = 10000, 4 seeds
    python scripts/kse_estimation/table_lorenz96.py --T 1000 --seeds 2  # quick check
    python scripts/kse_estimation/table_lorenz96.py --table-only        # rewrite the LaTeX from the cache

Outputs go to <out>/lorenz96.npz (cache) and <out>/lorenz96.tex.
'''

import argparse
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

from val.kse import build_attractor, make_evaluate
from val.kse.estimators import make_discrete_qr, make_discrete_riccati

DIMS = [10, 25, 50, 75, 100]

METHODS = {
    'Discrete Riccati': make_discrete_riccati,
    'Discrete QR':      make_discrete_qr,
}

## row label in the LaTeX table for each method
LABELS = {
    'Discrete Riccati': 'Ours',
    'Discrete QR':      'QR  ',
}

DEFAULT_OUT = Path(__file__).resolve().parents[2] / 'results' / 'kse_estimation'


def compute(T: float, seeds: int, dims: list[int]) -> np.ndarray:
    '''
    Final estimates of the positive sum, shape (len(dims), len(METHODS), seeds).
    '''

    results = np.full((len(dims), len(METHODS), seeds), np.nan)

    for i, N in enumerate(dims):
        attractor = build_attractor('lorenz96', N = N)
        steps = int(round(T / attractor.dt))
        x0 = jax.vmap(attractor.reset)(jax.random.split(jax.random.key(0), seeds))

        for j, make_estimator in enumerate(METHODS.values()):
            evaluate = make_evaluate(attractor, make_estimator(attractor), steps, record_every = steps)
            out = jax.jit(jax.vmap(evaluate))(x0)
            results[i, j] = np.asarray(out['kse'][:, -1])

        print(f'lorenz96 N={N}: ' + '  '.join(
            f'{m} {results[i, j].mean():.4f} +/- {results[i, j].std():.4f}' for j, m in enumerate(METHODS)),
            flush = True)

    return results


def to_text(results: np.ndarray, dims: list[int]) -> str:

    riccati, qr = results[:, 0], results[:, 1]
    header = f'{"N":>4}  {"Discrete Riccati":>22}  {"Discrete QR":>22}  {"|diff|":>10}  {"rel diff":>10}'
    rows = [header, '-' * len(header)]
    for i, N in enumerate(dims):
        diff = np.abs(riccati[i].mean() - qr[i].mean())
        rows.append(
            f'{N:>4}  {riccati[i].mean():>12.4f} +/- {riccati[i].std():<6.4f}  '
            f'{qr[i].mean():>12.4f} +/- {qr[i].std():<6.4f}  '
            f'{diff:>10.2e}  {diff / np.abs(qr[i].mean()):>10.2e}')
    return '\n'.join(rows)


def to_latex(results: np.ndarray, dims: list[int], seeds: int) -> str:
    '''
    Methods as rows, dimensions as columns, mean +/- std over seeds.
    '''

    rows = []
    for j, method in enumerate(METHODS):
        cells = [f'${results[i, j].mean():.2f} \\pm {results[i, j].std():.2f}$' for i in range(len(dims))]
        rows.append(f'    {LABELS[method]} & ' + ' & '.join(cells) + ' \\\\')

    return '\n'.join([
        '\\begin{table}[h]',
        '  \\centering',
        '  \\vspace{4pt}',
        '  {\\small',
        '  \\begin{tabular}{l' + 'c' * len(dims) + '}',
        '    \\toprule',
        '    Method & ' + ' & '.join(f'$N = {N}$' for N in dims) + ' \\\\',
        '    \\midrule',
        *rows,
        '    \\bottomrule',
        '  \\end{tabular}',
        '  }',
        '  \\vspace{8pt}',
        '  \\caption{Sum of positive Lyapunov exponents on Lorenz-96 for increasing state dimension $N$, '
        f'estimated with our recursion and with QR decomposition (mean $\\pm$ std over {seeds} random seeds).}}',
        '  \\label{tab:lorenz96}',
        '\\end{table}',
        '',
    ])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--T', type = float, default = 10_000.0, help = 'integration time per run (default: 10000)')
    parser.add_argument('--seeds', type = int, default = 4, help = 'number of initial conditions (default: 4)')
    parser.add_argument('--dims', type = int, nargs = '+', default = DIMS,
                        help = f'state dimensions N to evaluate (default: {DIMS})')
    parser.add_argument('--out', type = Path, default = DEFAULT_OUT,
                        help = f'output directory (default: {DEFAULT_OUT})')
    parser.add_argument('--table-only', action = 'store_true', help = 'rewrite the table from the cached results')
    args = parser.parse_args()

    args.out.mkdir(parents = True, exist_ok = True)
    cache = args.out / 'lorenz96.npz'

    if args.table_only:
        data = np.load(cache)
        results, T, seeds, dims = data['results'], float(data['T']), int(data['seeds']), list(data['dims'])
    else:
        results, T, seeds, dims = compute(args.T, args.seeds, args.dims), args.T, args.seeds, args.dims
        np.savez(cache, results = results, T = T, seeds = seeds, dims = dims, methods = list(METHODS))

    print()
    print(f'T = {T}, seeds = {seeds}')
    print(to_text(results, dims))

    out = args.out / 'lorenz96.tex'
    out.write_text(to_latex(results, dims, seeds))
    print(f'\nsaved {out}')
