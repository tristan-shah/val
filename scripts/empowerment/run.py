'''
Empowerment baseline (Figure 4, Table 1): at every step the agent applies the bang-bang action
along the gradient of empowerment, computed exactly under a linear-Gaussian channel approximation
of the dynamics over the planning horizon. Deterministic, so one run per environment.

    python scripts/empowerment/run.py --task CART_POLE

Writes traj.npy, empowerment.png and vid.mp4 to results/EMPOWERMENT/<TASK>/<run>.
Run from the repository root; set MUJOCO_GL=egl for offscreen rendering on a headless machine.
'''

from argparse import ArgumentParser
from pathlib import Path

from jax import Array
from jax import numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from val import Dynamics, make_step
from val.empowerment import compute_empowerment, compute_empowerment_grad

def render_cart_pole(dyn: Dynamics, X: Array, path: Path):
    dyn.render(X, path = path / 'vid.mp4', skip = 2, distance = 4)
    return None

def render_double_pendulum(dyn: Dynamics, X: Array, path: Path):
    dyn.render(X, path = path / 'vid.mp4', distance = 5.0, skip = 2, lookat = jnp.array([0, 0, 2.2]))
    return None

def render_triple_pendulum(dyn: Dynamics, X: Array, path: Path):
    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 5)
    return None

def render_humulum(dyn: Dynamics, X: Array, path: Path):
    dyn.render(X, path = path / 'vid.mp4', skip = 1, distance = 5, lookat = jnp.array([0.0, 0.0, 0.0]))
    return None

params = {
    'CART_POLE': {
        'horizon': 300,
        'steps': 1200,
        'dt': 0.01,
        'xml_path': 'xml/cart_pole.xml',
        'render': render_cart_pole
    },
    'DOUBLE_PENDULUM': {
        'horizon': 500,
        'steps': 2000,
        'dt': 0.01,
        'xml_path': 'xml/double_pendulum.xml',
        'render': render_double_pendulum
    },
    'TRIPLE_PENDULUM': {
        'horizon': 512,
        'steps': 2000,
        'dt': 0.01,
        'xml_path': 'xml/triple_pendulum.xml',
        'render': render_triple_pendulum,
        'init': lambda dyn: jnp.zeros(dyn.state_dim).at[0].set(jnp.pi),
        'setup': lambda dyn: dyn.mjx_model.replace(
            actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(25),
            dof_damping = dyn.mjx_model.dof_damping.at[:].set(3.5)
        )
    },
    'HUMULUM': {
        'horizon': 500,
        'steps': 1200,
        'dt': 0.01,
        'xml_path': 'xml/humulum.xml',
        'render': render_humulum,
        'init': lambda dyn: jnp.load('xml/hanging.npy'),
        'setup': lambda dyn: dyn.mjx_model.replace(
            actuator_gear = dyn.mjx_model.actuator_gear.at[:, 0].set(50),
            dof_damping = dyn.mjx_model.dof_damping.at[:].set(5.0)
        )
    }
}

if __name__ == '__main__':

    parser = ArgumentParser(description = __doc__)
    parser.add_argument('--task', type = str, default = 'CART_POLE', choices = list(params))
    parser.add_argument('--steps', type = int, default = None, help = 'episode length; defaults to the paper setting for the task')
    args = parser.parse_args()
    task = args.task

    horizon = params[task]['horizon']
    P = params[task]['horizon']       ## total control power equals the horizon (unit power per step)
    steps = params[task]['steps'] if args.steps is None else args.steps
    dt = params[task]['dt']

    name = f'h={horizon}-dt={dt}'
    root = Path(f'results/EMPOWERMENT/{task}')
    path = root / name
    path.mkdir(parents = True, exist_ok = True)

    dyn = Dynamics(path = params[task]['xml_path'], dt = params[task]['dt'])
    if 'setup' in params[task]:
        dyn.mjx_model = params[task]['setup'](dyn)
    print(f'Timestep = {dt}')
    print(f'Horizon = {horizon}')

    ## make the step function
    step = make_step(dyn)

    ## initialize state
    if 'init' in params[task]:
        xt = params[task]['init'](dyn)
    else:
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
        print(f'step {t:5d}  empowerment {float(e):.4f}', flush = True)

        ## propagate dynamics
        xt = step(xt, ut)

        ## log state
        X = X.at[t+1].set(xt)

    ## save trajectory data
    jnp.save(path / 'traj.npy', X)

    ## plotting the empowerment over time
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'{task} empowerment', fontsize = 14)
    ax.set_xlim(0.0, steps * dt)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.set_xlabel('Time (s)', fontsize = 14)
    ax.set_ylabel('Empowerment (Nats)',  fontsize = 14)
    times = jnp.linspace(0.0, X.shape[0] * dt, X.shape[0]-1)
    ax.plot(times, empowerment_hist)
    fig.tight_layout()
    fig.savefig(path / 'empowerment.png', dpi = 300)
    plt.close(fig)

    ## render an animation
    params[task]['render'](dyn, X, path)