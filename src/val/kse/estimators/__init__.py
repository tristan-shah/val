from val.kse.estimators.qr import make_discrete_qr
from val.kse.estimators.riccati import make_continuous_riccati, make_discrete_riccati

ESTIMATORS = {
    'discrete QR':          make_discrete_qr,
    'discrete Riccati':     make_discrete_riccati,
    'continuous Riccati':   make_continuous_riccati,
}

__all__ = [
    'ESTIMATORS',
    'make_continuous_riccati',
    'make_discrete_qr',
    'make_discrete_riccati',
]
