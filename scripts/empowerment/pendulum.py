from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.empowerment import compute_empowerment, compute_empowerment_grad

def render_pendulum(dyn: Dynamics, X: Array, path: Path):
    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 4)
    return None

def render_cart_pole(dyn: Dynamics, X: Array, path: Path):
    dyn.render(X, path = path / 'vid.mp4', skip = 2, distance = 4)
    return None

def render_double_pendulum(dyn: Dynamics, X: Array, path: Path):
    dyn.render(X, path = path / 'vid.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))
    return None

params = {
    'SINGLE_PENDULUM': {
        'horizon': 150,
        'steps': 600,
        'dt': 0.05,
        'xml_path': 'xml/pendulum.xml',
        'render': render_pendulum
    },
    'CART_POLE': {
        'horizon': 300,
        'steps': 1200,
        'dt': 0.01,
        'xml_path': 'xml/cart_pole.xml',
        'render': render_cart_pole
    },
    'DOUBLE_PENDULUM': {
        'horizon': 500,
        'steps': 3000, #1200,
        'dt': 0.01,
        'xml_path': 'xml/double_pendulum.xml',
        'render': render_double_pendulum
    }
}

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    # task = 'SINGLE_PENDULUM'
    # task = 'CART_POLE'
    task = 'DOUBLE_PENDULUM'

    horizon = params[task]['horizon']
    P = params[task]['horizon']
    steps = params[task]['steps']
    dt = params[task]['dt']

    name = f'h={horizon}-dt={dt}'
    root = Path(f'results/empowerment/{task}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    dyn = Dynamics(path = params[task]['xml_path'], dt = params[task]['dt'])
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    ## make the step function
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

    ## save trajectory data
    jnp.save(path / 'traj.npy', X)

    ## plotting the empowerment over time
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Single Pendulum Empowerment', fontsize = 14)
    ax.set_xlim(0.0, steps * dt)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.set_xlabel('Time (s)', fontsize = 14)
    ax.set_ylabel('Empowerment (Nats)',  fontsize = 14)
    times = jnp.linspace(0.0, X.shape[0] * dt, X.shape[0]-1)
    ax.plot(times, empowerment_hist)
    fig.tight_layout()
    fig.savefig(path / 'empowerment.png', dpi = 300)

    ## render an animation
    params[task]['render'](dyn, X, path)