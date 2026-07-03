"""
wandb-instrumented copy of triple_pendulum.py.

The inner problem is unchanged: CEM-MPC still optimizes the CIP objective. On top
of that we log the pendulum tip height *live* to Weights & Biases every MPC step,
and report summary height metrics that a W&B Sweep can optimize.

Standalone:
    python scripts/cip/experiments/triple_pendulum_wandb.py --horizon 128 --beta 0.0

Under a sweep (see height_sweep.yaml): wandb injects the hyperparameters via
wandb.config, overriding the argparse defaults below.

Note: unlike the original, MUJOCO_GL is not force-set to 'egl' (invalid on macOS).
Export it yourself if you need offscreen rendering, e.g. MUJOCO_GL=egl on a Linux
GPU box, or MUJOCO_GL=glfw on a mac. Height logging does not require any GL.
"""

from argparse import ArgumentParser
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import numpy as np
import mujoco
import jax
from jax import numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from val import Dynamics, make_step
from val.dynamics import split_state
from val.cem import CEM
from val.cip import make_compute_cip


def parse_args():
    p = ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--component", type=str, default="ol")
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--gear", type=float, default=25.0)
    p.add_argument("--damping", type=float, default=3.5)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--elite_frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--beta", type=float, default=2.5)
    p.add_argument("--project", type=str, default="triple-pendulum-height")
    p.add_argument("--entity", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # argparse values are the defaults; a W&B Sweep overrides them via wandb.config.
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config={
            "seed": args.seed,
            "component": args.component,
            "horizon": args.horizon,
            "shots": args.shots,
            "gear": args.gear,
            "damping": args.damping,
            "iterations": args.iterations,
            "elite_frac": args.elite_frac,
            "steps": args.steps,
            "beta": args.beta,
        },
    )
    cfg = wandb.config

    dt = 0.01
    smoothing = 0.1
    rho = 0.9

    key = jax.random.PRNGKey(cfg.seed)
    component = cfg.component

    name = (
        f"retest-seed={cfg.seed}-gear={cfg.gear}-damp={cfg.damping}-beta={cfg.beta}"
        f"-h={cfg.horizon}-shots={cfg.shots}-iter={cfg.iterations}"
        f"-elite={cfg.elite_frac}-smooth={smoothing}-rho={rho}-dt={dt}-steps={cfg.steps}"
    )
    path = Path(f"results/TRIPLE_PENDULUM/wandb/{component}") / name
    path.mkdir(parents=True, exist_ok=True)

    dyn = Dynamics("xml/triple_pendulum.xml", dt=dt)

    # override default gear strength
    dyn.mjx_model = dyn.mjx_model.replace(
        actuator_gear=dyn.mjx_model.actuator_gear.at[:, 0].set(cfg.gear)
    )
    if cfg.damping is not None:
        dyn.mjx_model = dyn.mjx_model.replace(
            dof_damping=dyn.mjx_model.dof_damping.at[:].set(cfg.damping)
        )

    step = make_step(dyn)
    compute_cip = make_compute_cip(dyn, component)

    def objective(xt, U):
        J, info = compute_cip(xt, U)
        control_penalty = jnp.mean(jnp.sum(U ** 2, axis=1))
        info["control_penalty"] = control_penalty
        return J - cfg.beta * control_penalty, info

    batch_objective = jax.jit(jax.vmap(objective, in_axes=(None, 0)))

    mpc = CEM(
        dyn,
        batch_objective,
        cfg.shots,
        cfg.horizon,
        cfg.iterations,
        cfg.elite_frac,
        smoothing,
        rho,
    )

    # --- live tip-height helper (same torso + [0,0,0.4] tip as the paper plots) ---
    model = dyn.model
    mjdata = mujoco.MjData(model)
    torso_id = model.body("torso").id
    tip_local = np.array([0.0, 0.0, 0.4])

    def tip_height(xt):
        qpos, qvel = split_state(np.array(xt), dyn.nq)
        mjdata.qpos[:] = qpos
        mjdata.qvel[:] = qvel
        mujoco.mj_forward(model, mjdata)
        xmat = mjdata.xmat[torso_id].reshape(3, 3)
        return float((mjdata.xpos[torso_id] + xmat @ tip_local)[2])

    xt = jnp.zeros(dyn.state_dim).at[0].set(jnp.pi)

    steps = int(cfg.steps)
    X = jnp.zeros((steps + 1, dyn.state_dim)).at[0].set(xt)
    hist = jnp.zeros((steps, 3))
    controls = jnp.zeros((steps, dyn.control_dim))
    heights = np.zeros(steps)

    max_h = -np.inf
    for t in range(steps):
        key, subkey = jax.random.split(key)
        ut, J, info, U = mpc(xt, subkey)
        xt = step(xt, ut)

        h = tip_height(xt)
        heights[t] = h
        max_h = max(max_h, h)

        controls = controls.at[t].set(ut)
        X = X.at[t + 1].set(xt)
        hist = hist.at[t].set(
            jnp.array([info["cip"], info["ol"], info["cl"]])
        )

        # live logging (only the info keys the MPC actually propagates)
        log = {"height": h, "height_running_max": max_h, "objective": float(J)}
        for k in ("cip", "ol", "cl", "control_penalty"):
            if k in info:
                log[k] = float(info[k])
        wandb.log(log, step=t)

    # --- summary metrics the sweep optimizes ---
    last10 = float(heights[int(0.9 * steps):].mean())
    summary = {
        "height_last10": last10,
        "height_peak": float(heights.max()),
        "height_mean": float(heights.mean()),
        "height_final": float(heights[-1]),
    }
    wandb.log(summary, step=steps - 1)
    wandb.summary.update(summary)

    # --- persist trajectories ---
    jnp.save(path / "hist.npy", hist)
    jnp.save(path / "traj.npy", X)
    jnp.save(path / "U.npy", controls)
    np.save(path / "heights.npy", heights)

    # height-vs-time figure -> wandb
    T = np.arange(steps) * dt
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tip height")
    ax.plot(T, heights, label="height")
    ax.axhline(last10, color="red", ls="--", lw=0.8, label=f"last-10% mean = {last10:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "height.png", dpi=200)
    wandb.log({"height_curve": wandb.Image(fig)})
    plt.close(fig)

    # optional render (needs a working GL backend; never fatal)
    try:
        dyn.render(X, path=path / "vid.mp4", skip=1, distance=5)
        wandb.log({"rollout": wandb.Video(str(path / "vid.mp4"))})
    except Exception as e:  # noqa: BLE001
        print(f"[render skipped] {e}")

    run.finish()


if __name__ == "__main__":
    main()
