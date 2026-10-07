'''
Estimators of the sum of positive Lyapunov exponents (Kolmogorov-Sinai entropy under Pesin's
conditions) on chaotic attractors. Used by scripts/kse_estimation to reproduce Figure 3 and
Table 2 of the paper.

    from val.kse import build_attractor, make_evaluate
    from val.kse.estimators import make_discrete_riccati, make_discrete_qr
'''

from val.kse.attractors import ATTRACTOR_REGISTRY, Attractor, build_attractor
from val.kse.runner import Estimator, make_evaluate

__all__ = [
    'ATTRACTOR_REGISTRY',
    'Attractor',
    'Estimator',
    'build_attractor',
    'make_evaluate',
]
