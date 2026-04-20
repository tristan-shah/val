from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from jax import numpy as jnp

from val import Dynamics
from val.dynamics import split_state


def compute_tip_heights(X, model, nq):
    """Return the z-height of the torso tip (top of torso geom) for each state in X."""
    data = mujoco.MjData(model)
    torso_id = model.body('torso').id
    torso_tip_local = np.array([0., 0., 0.4])  # top of torso geom in body frame

    heights = []
    for xt in X:
        data.qpos, data.qvel = split_state(np.array(xt), nq)
        mujoco.mj_forward(model, data)
        xmat = data.xmat[torso_id].reshape(3, 3)
        tip_z = (data.xpos[torso_id] + xmat @ torso_tip_local)[2]
        heights.append(tip_z)

    return np.array(heights)[:-1]  # all but last state


def compute_max_tip_height(dyn):
    """Maximum tip height: foot body is fixed (no soft weld), so the exact
    geometric maximum is achieved when all joints are 0 (fully upright).
    foot z=1.5 + leg=0.5 + thigh=0.45 + torso=0.4 = 2.85 m"""
    model = dyn.model
    data = mujoco.MjData(model)
    torso_id = model.body('torso').id
    torso_tip_local = np.array([0., 0., 0.4])

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    xmat = data.xmat[torso_id].reshape(3, 3)
    return (data.xpos[torso_id] + xmat @ torso_tip_local)[2]


if __name__ == '__main__':

    results_dir = Path('results/CIP/TRIPLE_PENDULUM/ol')
    suffix = 'gear=25.0-beta=2.5-h=128-shots=2048-iter=2-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200'

    dt = 0.01
    dyn = Dynamics('xml/triple_pendulum.xml', dt=dt)

    all_heights = []
    # seeds = list(range(8))
    seeds = [0, 1, 3, 4, 5, 6, 7]
    print(f'seeds: {seeds}')

    for seed in seeds:
        run_dir = results_dir / f'retest-seed={seed}-{suffix}'
        traj_path = run_dir / 'traj.npy'
        if not traj_path.exists():
            print(f'Missing: {traj_path}')
            continue

        X = jnp.load(traj_path)
        heights = compute_tip_heights(X, dyn.model, dyn.nq)
        print(f'seed={seed}  shape={heights.shape}')
        all_heights.append(heights)

    all_heights = np.array(all_heights)  # (n_seeds, T)
    mean = all_heights.mean(axis=0)
    std = all_heights.std(axis=0)
    t = np.arange(mean.shape[0]) * dt

    max_tip_height = compute_max_tip_height(dyn)
    print(f'max tip height: {max_tip_height:.4f} m')

    plt.figure()
    plt.axhline(max_tip_height, color='red', linestyle='dashed', linewidth=1.0, label='max height')

    plt.plot(t, mean, label='CIP')
    plt.fill_between(t, mean - std, mean + std, alpha=0.3)

    # for i, seed in enumerate(seeds):
    #     plt.plot(t, all_heights[i], linewidth=0.8, alpha=0.5, label=f'seed {seed}')

    plt.xlabel('Time (s)')
    plt.ylabel('Tip height (m)')
    plt.title('Triple pendulum')
    plt.legend()
    plt.tight_layout()
    plt.savefig('triple_pendulum_height.png', dpi=300)
    plt.show()
