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


if __name__ == '__main__':

    results_dir = Path('/Users/tristanshah/Desktop/code/val/results/HUMULUM/efficient/ol')
    suffix = 'beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200'

    dt = 0.01
    dyn = Dynamics('xml/humulum.xml', dt=dt)

    all_heights = []
    # for seed in range(11):

    # max_seed = 3
    # max_seed = 11
    # for seed in range(max_seed + 1):

    seeds = [0, 1, 2, 3, 6, 7, 8, 9] ## standing
    # seeds = [4, 5] ## crouching


    from val.cip import make_compute_cip
    compute_cip = make_compute_cip(dyn, 'ol')

    for seed in seeds:
        run_dir = results_dir / f'seed={seed}-{suffix}'
        traj_path = run_dir / 'traj.npy'
        if not traj_path.exists():
            print(f'Missing: {traj_path}')
            continue
        X = jnp.load(traj_path)
        U = jnp.load(run_dir / 'U.npy')

        _, info = compute_cip(X[600], U[600:])
        print(info)


    #     heights = compute_head_heights(X, dyn.model, dyn.nq)

    #     print(heights.shape)
    #     all_heights.append(heights)

    # all_heights = np.array(all_heights)  # (n_seeds, T)
    # mean = all_heights.mean(axis=0)
    # std = all_heights.std(axis=0)
    # t = np.arange(mean.shape[0]) * dt



    # # diayn_head_height = jnp.load('diayn_head_height.npy')
    # diayn_best_skill = jnp.load('best_skill_seeds.npy')

    # diayn_mean = diayn_best_skill.mean(axis = 0)
    # diayn_std = diayn_best_skill.std(axis = 0)


    # plt.figure()
    # plt.plot(t, mean, label='CIP')
    # plt.fill_between(t, mean - std, mean + std, alpha=0.3)

    # plt.plot(t, diayn_mean, label = 'DIAYN (best skill)')
    # plt.fill_between(t, diayn_mean - diayn_std, diayn_mean + diayn_std, alpha=0.3)

    # plt.xlabel('Time (s)')
    # plt.ylabel('Head height (m)')
    # plt.title(f'Humanoid head height over time ({len(seeds)} seeds)')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(results_dir / 'head_height_avg.png', dpi = 300)
    # plt.show()