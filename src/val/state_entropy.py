'''
Particle-based state-entropy planning objective (the "APT (MPC)" baseline of Table 1).

This is the Kozachenko-Leonenko k-nearest-neighbor estimator of differential entropy
(Kozachenko & Leonenko 1987; Singh et al. 2003), which APT (Liu & Abbeel 2021) uses as an RL
reward. Here it scores a single planning rollout: every state of the rollout is a particle
and the objective is the log of the mean distance to its k nearest neighbors, averaged over
the trajectory,

    H(X) = (1/T) sum_i log( 1 + (1/k) sum_{j in kNN(i)} ||x_i - x_j||_2 ).

Maximizing it rewards rollouts whose states spread out over the planning horizon.
'''

import jax
from jax import Array
from jax import numpy as jnp

from val.dynamics import make_unroll


def particle_entropy(traj: Array, k: int = 12):
    '''
    k-NN particle entropy of a single trajectory of shape (T, d).

    Implemented without `jax.lax.top_k`, which is very slow on the XLA CPU backend: squared
    pairwise distances come from the Gram matrix (no (T, T, d) intermediate) and the k smallest
    per row are extracted with k min-and-mask passes. Matches the top_k form to ~1e-16 on
    continuous trajectories, where pairwise distances are distinct. Rows with fewer than k
    distinct distances (exactly repeated states) degrade to 0 instead of leaking +inf.
    '''
    T = traj.shape[0]

    ## squared pairwise distances via the Gram matrix: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq = jnp.sum(traj ** 2, axis = -1)
    gram = traj @ traj.T
    d2 = jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    d2 = jnp.where(jnp.eye(T, dtype = bool), jnp.inf, d2)      ## mask self-distance

    ## sum the k smallest distances per row
    def body(carry, _):
        cur, acc = carry
        m = jnp.min(cur, axis = -1)
        acc = acc + jnp.sqrt(m)
        cur = jnp.where(cur <= m[:, None], jnp.inf, cur)
        return (cur, acc), None

    (_, acc), _ = jax.lax.scan(body, (d2, jnp.zeros(T)), None, length = k)
    acc = jnp.where(jnp.isfinite(acc), acc, 0.0)
    return jnp.mean(jnp.log(1.0 + acc / k))


def make_compute_state_entropy(step: callable, k: int = 12, subsample: int = 1):
    '''
    Builds a CEM-compatible objective from a jax `step` callable. Returns a jitted function
    (xt, U) -> (H, info) where H is the particle entropy of the rollout of U from xt and
    info = {'entropy': H}. `subsample` thins the rollout before the O(T^2) neighbor search.
    '''
    unroll = make_unroll(step)

    def compute_state_entropy(xt: Array, U: Array):
        X = unroll(xt, U)
        H = particle_entropy(X[::subsample], k)
        return H, {'entropy': H}

    return jax.jit(compute_state_entropy)
