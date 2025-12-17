import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics
from general_cost_iLQR import iLQR, make_compute_pendulum_state_cost, make_compute_pendulum_control_cost

class MPC:
    def __init__(self, ilqr: iLQR, horizon: int, key):

        self.ilqr = ilqr
        self.U = (jax.random.normal(key, (horizon, ilqr.du)) * 0.05).clip(ilqr.low, ilqr.high)

    def __call__(self, xt: Array):
        X = self.ilqr.unroll(xt, self.U)
        ## do one update of iLQR with the new control sequence
        X, U, J, alpha, empowerment = self.ilqr(X, self.U)

        ut = U[0]
        ## shift over the plan by one action
        U = jnp.roll(U, shift = -1, axis = 0)
        U = U.at[-1].set(U[-2])
        self.U = U
        return ut, J, empowerment

if __name__ == '__main__':

    key = jax.random.PRNGKey(0)
    integrator = 'euler'
    dt = 0.01
    steps = 900
    horizon = 400

    angle_penalty = 20.0
    velocity_penalty = 0.5


    ## instantiate dynamics
    dyn = Dynamics(path = 'xml/pendulum.xml', integrator = integrator, dt = dt)

    ## building Q
    Q = jnp.array([
        [angle_penalty, 0.0],
        [0.0, velocity_penalty]
    ])

    state_cost = make_compute_pendulum_state_cost(Q)
    control_cost = make_compute_pendulum_control_cost()

    ## instantiate iLQR
    ilqr = iLQR(dyn, state_cost, control_cost)

    xt = jnp.zeros(dyn.state_dim)
    theta_0 = 0.0
    xt = xt.at[0].set(theta_0)




    mpc = MPC(ilqr, horizon, key)

    X = jnp.zeros((steps+1, dyn.state_dim))
    X = X.at[0].set(xt)

    cost_hist = []
    empowerment_hist = []
    for t in range(steps):
        ut, J, empowerment = mpc(xt)
        xt = ilqr.step(xt, ut)
        print(t, xt, ut, jnp.sum(empowerment))

        X = X.at[t+1].set(xt)
        cost_hist.append(J)
        empowerment_hist.append(jnp.sum(empowerment))

    dyn.render(X, path = f'horizon={horizon}_mpc.mp4')
    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlabel('Timestep')
    ax[0].set_ylabel('Horizon Cost')
    ax[0].plot(cost_hist)

    ax[1].set_xlabel('Timestep')
    ax[1].set_ylabel('Horizon Empowerment')
    ax[1].plot(empowerment_hist)
    fig.tight_layout()
    fig.savefig(f'horizon={horizon}_mpc.png', dpi = 300)
    plt.show()