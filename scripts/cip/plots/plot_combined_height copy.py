from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from jax import numpy as jnp

from val import Dynamics
from val.dynamics import split_state


# ---------------------------------------------------------------------------
# Height helpers — one pair (heights, max_height) per environment
# ---------------------------------------------------------------------------

def cart_pole_tip_heights(X, model, nq):
    data = mujoco.MjData(model)
    pole_id = model.body('pole').id
    tip_local = np.array([0., 0., -1.0])
    heights = []
    for xt in X:
        data.qpos, data.qvel = split_state(np.array(xt), nq)
        mujoco.mj_forward(model, data)
        xmat = data.xmat[pole_id].reshape(3, 3)
        heights.append((data.xpos[pole_id] + xmat @ tip_local)[2])
    return np.array(heights)[:-1]

def cart_pole_max_tip_height(dyn):
    model = dyn.model
    data = mujoco.MjData(model)
    pole_id = model.body('pole').id
    tip_local = np.array([0., 0., -1.0])
    data.qpos[:] = 0.0
    data.qpos[model.joint('pole_hinge').qposadr] = np.pi
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    xmat = data.xmat[pole_id].reshape(3, 3)
    return (data.xpos[pole_id] + xmat @ tip_local)[2]


def double_pendulum_tip_heights(X, model, nq):
    data = mujoco.MjData(model)
    link_id = model.body('second_link').id
    tip_local = np.array([0., 0., -1.0])
    heights = []
    for xt in X:
        data.qpos, data.qvel = split_state(np.array(xt), nq)
        mujoco.mj_forward(model, data)
        xmat = data.xmat[link_id].reshape(3, 3)
        heights.append((data.xpos[link_id] + xmat @ tip_local)[2])
    return np.array(heights)[:-1]

def double_pendulum_max_tip_height(dyn):
    model = dyn.model
    data = mujoco.MjData(model)
    link_id = model.body('second_link').id
    tip_local = np.array([0., 0., -1.0])
    data.qpos[:] = 0.0
    data.qpos[model.joint('hinge1').qposadr] = np.pi
    data.qpos[model.joint('hinge2').qposadr] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    xmat = data.xmat[link_id].reshape(3, 3)
    return (data.xpos[link_id] + xmat @ tip_local)[2]


def triple_pendulum_tip_heights(X, model, nq):
    data = mujoco.MjData(model)
    torso_id = model.body('torso').id
    tip_local = np.array([0., 0., 0.4])
    heights = []
    for xt in X:
        data.qpos, data.qvel = split_state(np.array(xt), nq)
        mujoco.mj_forward(model, data)
        xmat = data.xmat[torso_id].reshape(3, 3)
        heights.append((data.xpos[torso_id] + xmat @ tip_local)[2])
    return np.array(heights)[:-1]

def triple_pendulum_max_tip_height(dyn):
    model = dyn.model
    data = mujoco.MjData(model)
    torso_id = model.body('torso').id
    tip_local = np.array([0., 0., 0.4])
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    xmat = data.xmat[torso_id].reshape(3, 3)
    return (data.xpos[torso_id] + xmat @ tip_local)[2]


def gibbon_head_heights(X, model, nq):
    data = mujoco.MjData(model)
    head_id = model.body('head').id
    heights = []
    for xt in X:
        data.qpos, data.qvel = split_state(np.array(xt), nq)
        mujoco.mj_forward(model, data)
        heights.append(data.xpos[head_id, 2])
    return np.array(heights)[:-1]

def gibbon_max_head_height(dyn, n_steps=5000):
    model = dyn.model
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    head_id = model.body('head').id
    max_z = data.xpos[head_id, 2]
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        max_z = max(max_z, data.xpos[head_id, 2])
    return max_z


# ---------------------------------------------------------------------------
# Per-environment config
# ---------------------------------------------------------------------------
ENVS = [
    dict(
        title      = 'Cart pole',
        xml        = 'xml/cart_pole.xml',
        results    = 'results/CIP/CART_POLE/ol',
        suffix     = 'beta=0.0-h=400-shots=512-iter=1-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=1200',
        seeds      = list(range(10)),
        run_fmt    = 'seed={seed}-{suffix}',
        dt         = 0.01,
        heights_fn = cart_pole_tip_heights,
        max_fn     = cart_pole_max_tip_height,
        baselines  = [
            dict(label='Empowerment', traj_path='results/empowerment/CART_POLE/h=300-dt=0.01/traj.npy', deterministic=True),
            dict(label='DADS',  path='results/DADS/CART_POLE/best_skill_seeds.npy'),
            dict(label='DIAYN', path='results/DIAYN/CART_POLE/extremity_height_cart_pole_diayn_1.npy'),
            dict(label='SMM',   path='results/SMM/CART_POLE/extremity_height_cart_pole_smm_1.npy'),
        ],
    ),
    dict(
        title      = 'Double pendulum',
        xml        = 'xml/double_pendulum.xml',
        results    = 'results/CIP/DOUBLE_PENDULUM/ol',
        suffix     = 'beta=0.0-h=512-shots=1024-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=2400',
        seeds      = list(range(10)),
        run_fmt    = 'seed={seed}-{suffix}',
        dt         = 0.01,
        xlim       = 20,
        heights_fn = double_pendulum_tip_heights,
        max_fn     = double_pendulum_max_tip_height,
        baselines  = [
            dict(label='Empowerment', traj_path='results/empowerment/DOUBLE_PENDULUM/h=500-dt=0.01/traj.npy', deterministic=True),
            dict(label='DADS',  path='results/DADS/DOUBLE_PENDULUM/best_skill_seeds.npy'),
            dict(label='DIAYN', path='results/DIAYN/DOUBLE_PENDULUM/extremity_height_double_pendulum_diayn_1.npy'),
            dict(label='SMM',   path='results/SMM/DOUBLE_PENDULUM/extremity_height_double_pendulum_smm_1.npy'),
        ],
    ),
    dict(
        title      = 'Triple pendulum',
        xml        = 'xml/triple_pendulum.xml',
        results    = 'results/CIP/TRIPLE_PENDULUM/ol',
        suffix     = 'gear=25.0-beta=2.5-h=128-shots=2048-iter=10-elite=0.1-smooth=0.1-rho=0.9-dt=0.01-steps=2400',
        seeds      = list(range(10)),
        run_fmt    = 'retest-seed={seed}-{suffix}',
        dt         = 0.01,
        xlim       = 20,
        heights_fn = triple_pendulum_tip_heights,
        max_fn     = triple_pendulum_max_tip_height,
        baselines  = [
            dict(label='Empowerment', traj_path='results/empowerment/TRIPLE_PENDULUM/h=128-dt=0.01/traj.npy', deterministic=True),
            # dict(label='DADS',  path='results/DADS/TRIPLE_PENDULUM/best_skill_seeds.npy'),
            dict(label='DIAYN', path='results/DIAYN/TRIPLE_PENDULUM/extremity_height_diayn_1.npy'),
            dict(label='SMM',   path='results/SMM/TRIPLE_PENDULUM/extremity_height_smm_1.npy'),
        ],
    ),
    dict(
        title      = 'Gibbon',
        xml        = 'xml/humulum.xml',
        results    = '/Users/tristanshah/Desktop/code/val/results/CIP/HUMULUM/efficient/ol',
        suffix     = 'beta=9.0-h=512-shots=1024-iter=1-elite=0.2-smooth=0.1-rho=0.9-dt=0.01-steps=1200',
        seeds      = list(range(10)),
        run_fmt    = 'seed={seed}-{suffix}',
        dt         = 0.01,
        heights_fn = gibbon_head_heights,
        max_fn     = gibbon_max_head_height,
        baselines  = [
            dict(label='Empowerment', traj_path='results/empowerment/HUMULUM/h=500-dt=0.01/traj.npy', deterministic=True),
            dict(label='DADS',  path='results/DADS/HUMULUM/best_skill_seeds.npy'),
            dict(label='DIAYN', path='results/DIAYN/HUMULUM/best_skill_seeds_diayn_1.npy'),
            dict(label='SMM',   path='results/SMM/HUMULUM/best_skill_seeds_smm_1.npy'),
        ],
    ),
]


def load_env(cfg):
    dyn = Dynamics(cfg['xml'], dt=cfg['dt'])
    results_dir = Path(cfg['results'])
    all_heights = []

    for seed in cfg['seeds']:
        run_dir = results_dir / cfg['run_fmt'].format(seed=seed, suffix=cfg['suffix'])
        traj_path = run_dir / 'traj.npy'
        if not traj_path.exists():
            print(f'  Missing: {traj_path}')
            continue
        X = jnp.load(traj_path)
        all_heights.append(cfg['heights_fn'](X, dyn.model, dyn.nq))

    if not all_heights:
        raise FileNotFoundError(f'No trajectories found for {cfg["title"]}')

    all_heights = np.array(all_heights)
    max_h = cfg['max_fn'](dyn)
    min_h = all_heights[:, 0].mean()
    return dyn, all_heights, min_h, max_h, cfg['dt']


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # -----------------------------------------------------------------------
    # Last-10% table
    # -----------------------------------------------------------------------
    COLUMNS = ['CIP (ours)', 'Empowerment', 'DIAYN', 'DADS', 'SMM']
    COL_W   = 13

    def last10(arr_1d):
        n = len(arr_1d)
        return float(np.mean(arr_1d[int(0.9 * n):]))

    rows = []
    for cfg in ENVS:
        print(f'  Computing table row for {cfg["title"]} ...')
        dyn = Dynamics(cfg['xml'], dt=cfg['dt'])
        results_dir = Path(cfg['results'])
        all_h = []
        for seed in cfg['seeds']:
            p = results_dir / cfg['run_fmt'].format(seed=seed, suffix=cfg['suffix']) / 'traj.npy'
            if p.exists():
                X = jnp.load(p)
                all_h.append(cfg['heights_fn'](X, dyn.model, dyn.nq))
        if not all_h:
            rows.append({**{c: None for c in COLUMNS}, '_title': cfg['title']})
            continue
        min_h = np.array(all_h)[:, 0].mean()
        max_h = cfg['max_fn'](dyn)
        normed_cip  = (np.array(all_h) - min_h) / (max_h - min_h)
        xlim_steps  = int(cfg.get('xlim', 12) / cfg['dt'])
        if normed_cip.shape[1] < xlim_steps:
            raise ValueError(
                f'{cfg["title"]}: CIP trajectory length ({normed_cip.shape[1]} steps) '
                f'is shorter than xlim ({xlim_steps} steps = {cfg.get("xlim", 12)}s)'
            )
        cip_score   = float(np.mean([last10(s[:xlim_steps]) for s in normed_cip]))

        row = {c: None for c in COLUMNS}
        row['_title'] = cfg['title']
        row['_min_h'] = min_h
        row['_max_h'] = max_h
        row['_heights_fn'] = cfg['heights_fn']
        row['_dt'] = cfg['dt']
        row['_dyn'] = dyn
        row['CIP (ours)'] = cip_score

        for bl in cfg.get('baselines', []):
            label = bl['label']
            if 'traj_path' in bl:
                p = Path(bl['traj_path'])
                if p.exists():
                    bl_X = jnp.load(p)
                    bl_h = np.array(cfg['heights_fn'](bl_X, dyn.model, dyn.nq))[:xlim_steps]
                    bl_normed = (bl_h - min_h) / (max_h - min_h)
                    row[label] = last10(bl_normed)
            else:
                p = Path(bl['path'])
                if p.exists():
                    bl_heights = np.array(jnp.load(p))
                    bl_normed  = (bl_heights - min_h) / (max_h - min_h)
                    row[label] = float(np.mean([last10(s[:xlim_steps]) for s in bl_normed]))
        rows.append(row)

    title_w = max(len(r['_title']) for r in rows) + 2
    hdr = f"{'Task':<{title_w}}" + ''.join(f"{c:>{COL_W}}" for c in COLUMNS)
    sep = '-' * len(hdr)
    print()
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        cells = []
        for c in COLUMNS:
            v = r.get(c)
            cells.append(f'{v:>{COL_W}.4f}' if v is not None else f"{'---':>{COL_W}}")
        print(f"{r['_title']:<{title_w}}" + ''.join(cells))
    print(sep)
    print()

    plt.rcParams.update({
        'font.size':        7,
        'axes.titlesize':   8,
        'axes.labelsize':   7,
        'xtick.labelsize':  6,
        'ytick.labelsize':  6,
        'legend.fontsize':  6,
        'lines.linewidth':  1.0,
    })

    BASELINE_COLORS = {
        'Empowerment': 'black',
        'DADS':        'C2',
        'DIAYN':       'C1',
        'SMM':         'C4',
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.8))

    for ax, cfg in zip(axes, ENVS):
        print(f'Loading {cfg["title"]} ...')
        dyn, all_heights, min_h, max_h, dt = load_env(cfg)

        # normalise to [0, 1]
        normed = (all_heights - min_h) / (max_h - min_h)
        mean   = normed.mean(axis=0)
        std    = normed.std(axis=0)
        t      = np.arange(mean.shape[0]) * dt

        ax.axhline(1.0, color='red', linestyle='dashed', linewidth=0.8, label='max')
        for bl in cfg.get('baselines', []):
            color = BASELINE_COLORS.get(bl['label'], 'gray')
            if 'traj_path' in bl:
                p = Path(bl['traj_path'])
                if not p.exists():
                    print(f'  Baseline not found (skipping): {p}')
                    continue
                bl_X = jnp.load(p)
                xlim_steps = int(cfg.get('xlim', 12) / cfg['dt'])
                bl_h = np.array(cfg['heights_fn'](bl_X, dyn.model, dyn.nq))[:xlim_steps]
                bl_normed = (bl_h - min_h) / (max_h - min_h)
                bl_t = np.arange(bl_normed.shape[0]) * cfg['dt']
                ax.plot(bl_t, bl_normed, label=bl['label'], color='black', linestyle='-')
            else:
                p = Path(bl['path'])
                if not p.exists():
                    print(f'  Baseline not found (skipping): {p}')
                    continue
                bl_heights = np.array(jnp.load(p))
                bl_normed  = (bl_heights - min_h) / (max_h - min_h)
                bl_normed  = bl_normed[:, :xlim_steps]
                bl_mean    = bl_normed.mean(axis=0)
                bl_std     = bl_normed.std(axis=0)
                bl_t       = np.arange(bl_normed.shape[1]) * cfg['dt']
                ax.plot(bl_t, bl_mean, label=bl['label'], color=color)
                if not bl.get('deterministic', False):
                    ax.fill_between(bl_t, bl_mean - bl_std, bl_mean + bl_std, alpha=0.3, color=color)

        ax.plot(t, mean, label='CIP (ours)', color='C0', zorder=3)
        ax.fill_between(t, mean - std, mean + std, alpha=0.3, color='C0', zorder=3)

        ax.set_title(cfg['title'])
        ax.set_xlabel('Time (s)')
        _xlim = cfg.get('xlim', 12)
        ax.set_xlim(0, _xlim)
        ax.set_xticks([0, _xlim / 2, _xlim])
        ax.set_ylim(-0.05, 1.15)

    axes[0].set_ylabel('Normalised height')
    for ax in axes[1:]:
        ax.set_yticklabels([])

    # single shared legend beneath all panels, merging handles across all axes
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    order = ['max', 'CIP (ours)', 'Empowerment']
    ordered_handles, ordered_labels = [], []
    for name in order:
        if name in labels:
            idx = labels.index(name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    for i, l in enumerate(labels):
        if l not in order:
            ordered_handles.append(handles[i])
            ordered_labels.append(l)
    handles, labels = ordered_handles, ordered_labels
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8)

    fig.tight_layout(pad=0.5, w_pad=0.8)
    fig.savefig('combined_height.pdf', bbox_inches='tight')
    fig.savefig('combined_height.png', dpi=600, bbox_inches='tight')
    # plt.show()
