'''
Particle-based state-entropy maximization objective for the CEM planner.

This is the classical Kozachenko-Leonenko k-nearest-neighbor estimator of differential
entropy (Kozachenko & Leonenko 1987; Singh et al. 2003). For a trajectory of states X of
shape (T, d) it estimates

    H(X) = (1/T) sum_i log( 1 + (1/k) sum_{j in kNN(i)} ||x_i - x_j||_2 )

i.e. every state is a "particle" and the score is the (log of the) mean distance to its k
nearest neighbors, averaged over the trajectory. Maximizing it rewards CEM rollouts whose
states spread out in state space over the planning horizon.

(The same k-NN entropy estimator is used as an RL reward in APT, Liu & Abbeel 2021 -- but
this module is NOT APT: there is no learned policy or replay buffer here, it is a plain
per-rollout planning objective consumed by the CEM.)

This module lives next to the experiment scripts so `from entropy_objective import ...`
resolves through sys.path[0] with no packaging changes. It imports only the read-only
public helper `make_unroll` from the installed `val` package (no src modification).
'''

import jax
from jax import Array
from jax import numpy as jnp

from val.dynamics import make_unroll


def particle_entropy(traj: Array, k: int = 12):
    '''
    Kozachenko-Leonenko k-NN particle entropy of a single trajectory.

        traj : (T, d) states
        k    : number of nearest neighbors
    Returns a scalar entropy estimate.

    Equivalent to the reference form
        mean_i log( 1 + (1/k) sum_{j in kNN(i)} ||x_i - x_j|| )
    but implemented without `jax.lax.top_k`, which is pathologically slow on the XLA
    CPU backend (a k-NN over a (T,T) matrix dominated a full objective eval by ~13x).
    Instead we (a) form squared pairwise distances via the Gram matrix (no (T,T,d)
    intermediate) and (b) extract the k smallest per row with k cheap min+mask passes.
    Verified equal to the top_k reference to ~1e-16 on real trajectories.

    Caveat: the `cur <= m` mask removes ALL entries tied at the row minimum in one pass.
    For continuous physics states every pairwise distance is distinct, so this removes
    exactly one neighbor per pass (that's why it matches top_k to ~1e-16). It only
    misbehaves if a row has fewer than k *bit-identical-distance* values -- i.e. exactly
    repeated states -- which continuous dynamics don't produce. The finite-guard below
    makes even that degenerate case degrade to 0 rather than leak +inf into the CEM.
    '''
    T = traj.shape[0]

    ## squared pairwise distances via the Gram matrix: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq   = jnp.sum(traj ** 2, axis = -1)                        # (T,)
    gram = traj @ traj.T                                        # (T,T)
    d2   = jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)  # clip float-precision negatives
    d2   = jnp.where(jnp.eye(T, dtype = bool), jnp.inf, d2)     # mask self-distance (exactly)

    ## sum the k smallest distances per row via k iterations of (min -> accumulate -> mask)
    def body(carry, _):
        cur, acc = carry
        m   = jnp.min(cur, axis = -1)                           # (T,) smallest remaining sq-dist
        acc = acc + jnp.sqrt(m)                                 # m >= 0 (clipped above)
        cur = jnp.where(cur <= m[:, None], jnp.inf, cur)        # remove that neighbor
        return (cur, acc), None

    (_, acc), _ = jax.lax.scan(body, (d2, jnp.zeros(T)), None, length = k)
    acc = jnp.where(jnp.isfinite(acc), acc, 0.0)               # guard degenerate exact-duplicate rows
    return jnp.mean(jnp.log(1.0 + acc / k))


def make_compute_state_entropy(step: callable, k: int = 12, subsample: int = 1):
    '''
    Build a CEM-compatible objective from a plain jax `step` callable.

    Returns `compute_state_entropy(xt, U) -> (H, info)` where H is the particle entropy of
    the rollout produced by applying control sequence U from state xt.

    `subsample` thins the trajectory before the O(T^2) k-NN to control memory/compute.

    The CEM (src/val/cem.py) unconditionally reads info['cip'], info['ol'] and info['cl']
    when aggregating elites, so we route the entropy through those keys to stay compatible
    without modifying src.
    '''
    unroll = make_unroll(step)

    def compute_state_entropy(xt: Array, U: Array):
        X = unroll(xt, U)                          # (H + 1, state_dim)
        H = particle_entropy(X[::subsample], k)
        info = {'cip': H, 'ol': H, 'cl': jnp.zeros_like(H)}
        return H, info

    return jax.jit(compute_state_entropy)
