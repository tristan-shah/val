import jax
## match the precision setting used by the mujoco Dynamics (see val.dynamics)
jax.config.update('jax_enable_x64', True)
from jax import Array
from jax import numpy as jnp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio


class BallInBox:
    '''
    A very simple, pure-jax environment: a point-mass ball free-floating inside a
    2D box whose walls reflect the ball. No gravity, no drag -- the only forces are
    the 2D control thruster and wall reflections.

        state   xt = [px, py, vx, vy]
        control ut = [fx, fy]

    The wall reflections are parameterized by a restitution coefficient `restitution`:
        restitution = 1.0  -> perfect elastic bounce
        restitution < 1.0  -> the bounce absorbs some of the ball's velocity

    Exposes the same public surface the CEM planner / CIP builder consume from the
    mujoco `Dynamics` (state_dim, control_dim, dt, step, linearize, low/high, render).
    '''

    def __init__(
            self,
            dt: float = 0.05,
            bounds: tuple = (-1.0, 1.0, -1.0, 1.0),  ## (xmin, xmax, ymin, ymax)
            mass: float = 1.0,
            restitution: float = 1.0,
            force_limit: float = 1.0):

        self.dt = dt
        self.bounds = bounds
        self.mass = mass
        self.restitution = restitution
        self.force_limit = force_limit

        self.state_dim = 4
        self.control_dim = 2

        ## control ranges consumed by CEM (custom-env branch, no mjx_model)
        self.low = -force_limit * jnp.ones(self.control_dim)
        self.high = +force_limit * jnp.ones(self.control_dim)

        ## jit the step and its jacobian (mirrors Dynamics)
        self.step = jax.jit(self._step)
        self.linearize = jax.jit(jax.jacfwd(self.step, argnums = (0, 1)))

    @staticmethod
    def _reflect(p: Array, v: Array, lo: float, hi: float, e: float):
        '''
        Branch-free 1D reflection off the [lo, hi] walls. Position is mirrored back
        inside the box and velocity is flipped and scaled by restitution `e`.
        Assumes at most one wall-crossing per step (true for small dt / bounded speed).
        '''
        below = p < lo
        p = jnp.where(below, 2.0 * lo - p, p)
        v = jnp.where(below, -e * v, v)

        above = p > hi
        p = jnp.where(above, 2.0 * hi - p, p)
        v = jnp.where(above, -e * v, v)

        return p, v

    def _step(self, xt: Array, ut: Array):
        xmin, xmax, ymin, ymax = self.bounds
        e = self.restitution

        p = xt[:2]
        v = xt[2:]

        ## semi-implicit euler: integrate velocity, then position
        v = v + self.dt * (ut / self.mass)
        p = p + self.dt * v

        ## reflect independently on each axis
        px, vx = self._reflect(p[0], v[0], xmin, xmax, e)
        py, vy = self._reflect(p[1], v[1], ymin, ymax, e)

        return jnp.array([px, py, vx, vy])

    def render(
            self,
            X: Array,
            path: str,
            skip: int = 1,
            trail: int = 40,
            fps: int = 60):
        '''
        Render the trajectory to an mp4 with matplotlib (no mujoco). Draws the box,
        the ball, and a short fading position trail.
        '''
        xmin, xmax, ymin, ymax = self.bounds
        X = np.asarray(X)

        writer = imageio.get_writer(path, fps = fps)

        fig, ax = plt.subplots(1, 1, figsize = (5, 5), dpi = 150)

        for t in range(0, X.shape[0], skip):
            ax.clear()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            ## box outline
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                color = 'black', linewidth = 2)

            ## fading trail
            lo = max(0, t - trail)
            if t > lo:
                seg = X[lo:t + 1]
                ax.plot(seg[:, 0], seg[:, 1], color = 'tab:blue', alpha = 0.4, linewidth = 1)

            ## ball
            ax.plot(X[t, 0], X[t, 1], 'o', color = 'tab:red', markersize = 14)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)

        plt.close(fig)
        writer.close()
        return None
