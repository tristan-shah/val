from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from jax import numpy as jnp

from val import Dynamics
from val.dynamics import split_state


def compute_head_heights(X, model, nq):
    data = mujoco.MjData(model)
    head_id = model.body('head').id
    heights = []
    for xt in X:
        data.qpos, data.qvel = split_state(np.array(xt), nq)
        mujoco.mj_forward(model, data)
        heights.append(data.xpos[head_id, 2])
    # return np.array(heights)
    return np.array(heights)[:-1] ## all but last state


def compute_max_head_height(dyn, n_steps=5000):
    """Simulate the model from a fully upright qpos=0 pose and step forward
    until the soft weld constraints settle, then return the peak head z seen."""
    model = dyn.model
    data = mujoco.MjData(model)

    # fully upright: all joints at 0, zero velocity
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    head_id = model.body('head').id
    max_z = data.xpos[head_id, 2]
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        max_z = max(max_z, data.xpos[head_id, 2])

    return max_z


if __name__ == '__main__':

    results_dir = Path('/Users/tristanshah/Desktop/code/val/results/CIP/HUMULUM/efficient/ol')
    suffix = 'beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200'

    dt = 0.01
    dyn = Dynamics('xml/humulum.xml', dt=dt)

    all_heights = []

    seeds = [0, 1, 2, 3, 6, 7, 8, 9] ## standing
    # seeds = [4, 5] ## crouching
    # seeds = list(range(10))
    print(seeds)

    # from val.cip import make_compute_cip
    # compute_cip = make_compute_cip(dyn, 'ol')

    for seed in seeds:
        run_dir = results_dir / f'seed={seed}-{suffix}'
        traj_path = run_dir / 'traj.npy'
        if not traj_path.exists():
            print(f'Missing: {traj_path}')
            continue


        X = jnp.load(traj_path)
        U = jnp.load(run_dir / 'U.npy')

        # _, info = compute_cip(X[600], U[600:])
        # print(info)

        heights = compute_head_heights(X, dyn.model, dyn.nq)

        print(heights.shape)
        all_heights.append(heights)

    all_heights = np.array(all_heights)  # (n_seeds, T)
    mean = all_heights.mean(axis=0)
    std = all_heights.std(axis=0)
    t = np.arange(mean.shape[0]) * dt

    dads_best_skill = jnp.load('results/DADS/HUMULUM/best_skill_seeds.npy')
    print(dads_best_skill.shape)

    diayn_best_skill = jnp.load('results/DIAYN/HUMULUM/best_skill_seeds_diayn_1.npy')
    print(diayn_best_skill.shape)

    smm_best_skill = jnp.load('results/SMM/HUMULUM/best_skill_seeds_smm_1.npy')
    print(smm_best_skill.shape)

    diayn_mean = diayn_best_skill.mean(axis = 0)
    diayn_std = diayn_best_skill.std(axis = 0)

    smm_mean = smm_best_skill.mean(axis = 0)
    smm_std = smm_best_skill.std(axis = 0)

    dads_mean = dads_best_skill.mean(axis = 0)
    dads_std = dads_best_skill.std(axis = 0)


    # Maximum head height: simulate the model held in a fully upright pose and
    # run forward dynamics to settle the soft weld constraints, then read head z.
    # (Hard-coding the rigid-geometry bound of 1.20 m is not exact because the
    # weld has solimp=".9 .9 0.01" — feet can deviate slightly under load.)
    max_head_height = compute_max_head_height(dyn)

    plt.figure()
    plt.axhline(max_head_height, color='red', linestyle='dashed', linewidth=1.0, label='max height')
    plt.plot(t, mean, label='CIP')
    plt.fill_between(t, mean - std, mean + std, alpha=0.3)

    plt.plot(t, diayn_mean, label = 'DIAYN')
    plt.fill_between(t, diayn_mean - diayn_std, diayn_mean + diayn_std, alpha=0.3)

    plt.plot(t, smm_mean, label = 'SMM')
    plt.fill_between(t, smm_mean - smm_std, smm_mean + smm_std, alpha=0.3)

    plt.plot(t, dads_mean, label = 'DADS')
    plt.fill_between(t, dads_mean - dads_std, dads_mean + dads_std, alpha=0.3)

    plt.xlabel('Time (s)')
    # plt.ylabel('Head height (m)')
    # plt.title(f'Humanoid head height over time ({len(seeds)} seeds)')
    plt.ylabel('Extremity height (m)')
    plt.title('Gibbon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('head_height_avg.png', dpi = 300)
    plt.show()