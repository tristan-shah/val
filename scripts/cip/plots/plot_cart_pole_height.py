from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from jax import numpy as jnp

from val import Dynamics
from val.dynamics import split_state
from val.utils import smooth_angle_wrap


def compute_tip_heights(X, model, nq):
    """Return the z-height of the pole tip (end of pole geom) for each state in X."""
    data = mujoco.MjData(model)
    pole_id = model.body('pole').id
    pole_tip_local = np.array([0., 0., -1.0])  # end of pole geom in body frame

    heights = []
    for xt in X:
        data.qpos, data.qvel = split_state(np.array(xt), nq)
        mujoco.mj_forward(model, data)
        xmat = data.xmat[pole_id].reshape(3, 3)
        tip_z = (data.xpos[pole_id] + xmat @ pole_tip_local)[2]
        heights.append(tip_z)

    return np.array(heights)[:-1]  # all but last state


def compute_max_tip_height(dyn):
    """Maximum tip height: pole fully upright (hinge = pi).
    cart z=1.1 + pole body offset z=0.05 + pole length 1.0 = 2.15 m (exact, no soft constraints)."""
    model = dyn.model
    data = mujoco.MjData(model)
    pole_id = model.body('pole').id
    pole_tip_local = np.array([0., 0., -1.0])

    data.qpos[:] = 0.0
    data.qpos[model.joint('pole_hinge').qposadr] = np.pi  # fully upright
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    xmat = data.xmat[pole_id].reshape(3, 3)
    return (data.xpos[pole_id] + xmat @ pole_tip_local)[2]


def report(name, heights_2d, min_h, max_h, last_frac=0.10):
    T = heights_2d.shape[1]
    idx = int(T * last_frac)
    per_seed = (heights_2d[:, -idx:].mean(axis=1) - min_h) / (max_h - min_h)
    print(f'{name}')
    print(f'  normalized height (last {last_frac*100:.0f}% of traj), per seed: {np.round(per_seed, 4)}')
    print(f'  mean ± std: {per_seed.mean():.4f} ± {per_seed.std():.4f}\n')


if __name__ == '__main__':

    results_dir = Path('results/CIP/CART_POLE/ol')
    suffix = 'beta=0.0-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200'

    dt = 0.01
    dyn = Dynamics('xml/cart_pole.xml', dt=dt)

    all_heights = []
    seeds = list(range(10))
    loaded_seeds = []
    print(f'seeds: {seeds}')

    for seed in seeds:
        run_dir = results_dir / f'seed={seed}-{suffix}'
        traj_path = run_dir / 'traj.npy'
        if not traj_path.exists():
            print(f'Missing: {traj_path}')
            continue

        X = jnp.load(traj_path)
        # plt.plot(X[:, 1])

    # plt.show()
        heights = compute_tip_heights(X, dyn.model, dyn.nq)
        print(f'seed={seed}  shape={heights.shape}')
        all_heights.append(heights)
        loaded_seeds.append(seed)

    if not all_heights:
        raise FileNotFoundError(f'No trajectories found in {results_dir} with suffix "{suffix}"')

    all_heights = np.array(all_heights)  # (n_seeds, T)
    mean = all_heights.mean(axis=0)
    std = all_heights.std(axis=0)
    t = np.arange(mean.shape[0]) * dt

    max_tip_height = compute_max_tip_height(dyn)
    min_tip_height = all_heights[:, 0].mean()  # mean height at t=0 across seeds

    print(f'min_tip_height (t=0 mean): {min_tip_height:.4f} m')
    print(f'max_tip_height (upright):  {max_tip_height:.4f} m\n')

    report('CIP', all_heights, min_tip_height, max_tip_height)

    plt.figure()
    plt.axhline(max_tip_height, color='red', linestyle='dashed', linewidth=1.0, label='max height')
    # for i, seed in enumerate(loaded_seeds):
    #     plt.plot(t, all_heights[i], linewidth=0.8, alpha=0.5, label=f'seed {seed}')
    
    plt.plot(t, mean, label='mean')
    plt.fill_between(t, mean - std, mean + std, alpha=0.3)

    plt.xlabel('Time (s)')
    plt.ylabel('Pole tip height (m)')
    plt.title('Cart pole')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cart_pole_height.png', dpi=300)
    plt.show()
