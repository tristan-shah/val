import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.empowerment import compute_empowerment, compute_empowerment_grad

# from soc_emp import Dynamics
# from soc_emp.empowerment import compute_empowerment, compute_empowerment_grad

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    ## simulation horizon
    horizon = 300
    P = horizon
    steps = 600

    ## load in xml
    xml_path = 'xml/pendulum.xml'
    # xml_path = 'xml/double_pendulum.xml'
    dyn = Dynamics(path = xml_path, dt = 0.05)
    # dyn = Dynamics(path = xml_path, dt = 0.01)
    dt = dyn.mjx_model.opt.timestep
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    step = make_step(dyn)

    ## initialize state
    xt = dyn.init_state()

    ## tensor for state storage
    X = jnp.zeros((steps + 1, dyn.state_dim))
    X = X.at[0].set(xt)

    ## zero control planning horizon
    U = jnp.zeros((horizon, dyn.control_dim))

    empowerment_hist = []
    for t in range(steps):

        _, B = dyn.linearize(xt, jnp.zeros(dyn.control_dim)) ## obtain control gain
        grad_E = compute_empowerment_grad(dyn, xt, U, P) ## compute gradient of empowerment

        ## bang bang control
        ut = B.T @ grad_E
        ut = jnp.sign(ut) * 1.0
        ut = ut.at[ut == 0].set(1.0)

        e = compute_empowerment(dyn, xt, U, P)
        empowerment_hist.append(e)
        print(t, xt, ut, e)

        ## propagate dynamics
        xt = step(xt, ut)

        ## log state
        X = X.at[t+1].set(xt)

    jnp.save(f'EMPOWERMENT-single_pendulum/empowerment-h={horizon}-traj.npy', X)

    # times = jnp.linspace(0.0, X.shape[0] * dt, X.shape[0]-1)

    # ## plotting the empowerment over time
    # fig, ax = plt.subplots(1, 1)
    # fig.suptitle('Single Pendulum Empowerment', fontsize = 14)
    # ax.set_xlim(0.0, steps * dt)
    # ax.tick_params(axis = 'both', labelsize = 12)
    # ax.set_xlabel('Time (s)', fontsize = 14)
    # ax.set_ylabel('Empowerment (Nats)',  fontsize = 14)
    # ax.plot(times, empowerment_hist)
    # fig.tight_layout()
    # fig.savefig(f'pendulum_empowerment.png', dpi = 300)

    # ## render an animation
    # dyn.render(X, path = 'pendulum_empowerment.mp4', skip = 1)
    # # dyn.render(X, path = 'double_pendulum_empowerment.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))
