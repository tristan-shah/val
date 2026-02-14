import jax
from jax import Array
from jax import numpy as jnp
import colorednoise

from val import Dynamics
from val.utils import ar1_noise

class CEM:
    def __init__(
            self, 
            dyn: Dynamics, 
            objective: callable,
            shots: int, 
            horizon: int, 
            iterations: int, 
            elite_frac: float,
            smoothing: float,
            rho: float = 0.9):
        
        assert 0.0 < elite_frac <= 1.0

        ## actuator ranges
        self.low = dyn.mjx_model.actuator_ctrlrange[:, 0]
        self.high = dyn.mjx_model.actuator_ctrlrange[:, 1]

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

        ## set a minimum exploration amount
        self.min_std = 0.05 * (self.high - self.low) ## default

        ## initial mean and std
        self.mean = jnp.zeros((self.horizon, self.control_dim))
        self.std = jnp.ones((self.horizon, self.control_dim))

        ## store the elite action sequences
        self.elites = None

    # def roll(self, key):
    def roll(self):

        ## roll backward
        self.mean = jnp.roll(self.mean, shift = -1, axis = 0)
        self.std = jnp.roll(self.std, shift = -1, axis = 0)
        # self.elites = jnp.roll(self.elites, shift = -1, axis = 1)

        ## set last element
        self.mean = self.mean.at[-1].set(self.mean[-2])
        self.std = self.std.at[-1].set(self.std[-2])

        # if self.elites is not None:
        #     ## add a random last action to the shifted elites
        #     key, subkey = jax.random.split(key)
        #     random_last = jax.random.uniform(subkey, (self.shots, self.control_dim), minval = self.low, maxval = self.high)
        #     self.elites = self.elites.at[:, -1, :].set(random_last)

        return None

    def __call__(self, xt: Array, key):

        for _ in range(self.iterations):

            key, subkey = jax.random.split(key)
            ## generate correlated noise
            noise = ar1_noise(subkey, self.shots, self.horizon, self.control_dim, self.rho)
            # noise = colorednoise.powerlaw_psd_gaussian(1.0, size = (self.shots, self.control_dim, self.horizon))
            # noise = jnp.permute_dims(jnp.array(noise), (0, 2, 1))

            ## generate a batch of random control signals
            U_batch = self.mean[None, :, :] + self.std[None, :, :] * noise
            ## clip the sampled sequences within the allowable range
            U_batch = U_batch.clip(self.low[None, None, :], self.high[None, None, :])
            ## evaluate control signals in parallel
            J = self.objective(xt, U_batch)

            ## select top performing control sequences
            elite_idx = jnp.argsort(J, descending = True)[:self.n_elite]
            U_elite = U_batch[elite_idx]
            # ## store elites from previous iteration
            # self.elites = U_elite

            ## fit gaussian
            new_mean = jnp.mean(U_elite, axis = 0)
            new_std = jnp.std(U_elite, axis = 0)

            ## smooth update
            self.mean = (1 - self.smoothing) * new_mean + self.smoothing * self.mean
            self.std  = (1 - self.smoothing) * new_std  + self.smoothing * self.std
            ## maintain minimum exploration
            self.std = self.std.clip(min = self.min_std)

        ## return best action (that was actually tested)
        best_idx = jnp.argmax(J)
        U_best = U_batch[best_idx]
        ut = U_best[0]

        ## rolls over mean, std, and elite sequences one step
        # self.roll(key)
        self.roll()

        return ut, J[elite_idx].mean()
