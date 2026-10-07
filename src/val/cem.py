import jax
from jax import Array
from jax import numpy as jnp

from val.utils import ar1_noise


class CEM:
    '''
    Improved Cross-Entropy Method planner (Pinneri et al. 2021) used for MPC.

    `dyn` supplies the actuator ranges: a MuJoCo `Dynamics` through its mjx model, or a
    custom environment (e.g. `BallInBox`) through plain `low` / `high` arrays.
    `objective(xt, U_batch) -> (J, info)` scores a batch of control sequences; every entry of
    `info` is averaged over the elites and returned alongside the chosen action.
    '''
    def __init__(
            self,
            dyn,
            objective: callable,
            shots: int,
            horizon: int,
            iterations: int,
            elite_frac: float,
            smoothing: float,
            rho: float = 0.9):

        assert 0.0 < elite_frac <= 1.0

        ## actuator ranges
        if hasattr(dyn, 'mjx_model'):
            self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
            self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]
        else:
            self.low, self.high = dyn.low, dyn.high

        ## objective function
        self.objective = objective

        ## planning parameters
        self.shots = shots
        self.horizon = horizon
        self.control_dim = dyn.control_dim
        self.iterations = iterations
        self.n_elite = max(1, int(elite_frac * shots))
        self.smoothing = smoothing
        self.rho = rho

        ## minimum exploration amount
        self.min_std = 0.05 * (self.high - self.low)

        ## initial sampling distribution
        self.mean = jnp.zeros((self.horizon, self.control_dim))
        self.std = jnp.ones((self.horizon, self.control_dim))

    def roll(self):
        '''
        Shifts the sampling distribution one step forward in time, repeating the last entry.
        '''
        self.mean = jnp.roll(self.mean, shift = -1, axis = 0)
        self.std = jnp.roll(self.std, shift = -1, axis = 0)
        self.mean = self.mean.at[-1].set(self.mean[-2])
        self.std = self.std.at[-1].set(self.std[-2])
        return None

    def __call__(self, xt: Array, key, roll: bool = True):
        '''
        Plans from state xt. Returns (ut, J_elite, info) where ut is the first action of the
        best sampled sequence, J_elite the mean elite objective and info the elite-averaged
        objective diagnostics.
        '''

        for _ in range(self.iterations):

            key, subkey = jax.random.split(key)
            ## generate correlated noise
            noise = ar1_noise(subkey, self.shots, self.horizon, self.control_dim, self.rho)

            ## sample a batch of control sequences and clip to the actuator range
            U_batch = self.mean[None, :, :] + self.std[None, :, :] * noise
            U_batch = U_batch.clip(self.low[None, None, :], self.high[None, None, :])

            ## evaluate control sequences in parallel
            J, info = self.objective(xt, U_batch)

            ## select the top performing control sequences
            elite_idx = jnp.argsort(J, descending = True)[:self.n_elite]
            U_elite = U_batch[elite_idx]

            ## fit a gaussian to the elites with a smoothed update
            new_mean = jnp.mean(U_elite, axis = 0)
            new_std = jnp.std(U_elite, axis = 0)
            self.mean = (1 - self.smoothing) * new_mean + self.smoothing * self.mean
            self.std = (1 - self.smoothing) * new_std + self.smoothing * self.std

            ## maintain minimum exploration
            self.std = self.std.clip(min = self.min_std)

        ## return the first action of the best sequence that was actually evaluated
        ut = U_batch[jnp.argmax(J)][0]

        if roll:
            self.roll()

        info = {k: v[elite_idx].mean() for k, v in info.items()}

        return ut, J[elite_idx].mean(), info
