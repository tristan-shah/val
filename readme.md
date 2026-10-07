# Emergence of Physical Intelligence via Controllable Information Production

Code for the paper. Controllable Information Production (CIP) is the open-loop information production rate of a controlled dynamical system; maximizing it with model predictive control (MPC) drives self-righting and upright stabilization on the cart pole, double and triple pendulum, and a hanging planar humanoid (the "gibbon"), with no external reward.

## Setup

```bash
git clone --recursive https://github.com/tristan-shah/val.git && cd val
conda create -n val python=3.10
conda activate val
pip install -e .
pip install "jax[cuda12]"        # GPU build of JAX, used for the MPC experiments
```

Run every command from the repository root. On a headless Linux machine set `MUJOCO_GL=egl` so videos render offscreen.

## Running

Experiments write to `results/<METHOD>/<ENVIRONMENT>/<run>/` with `traj.npy`, `U.npy`, `hist.npy` and a video. Figures are written to `figures/`. Script defaults are the paper's hyperparameters (Table 3), so a run needs only a seed.

**CIP (Section 4).** The paper uses seeds 0 to 9.

```bash
python scripts/cip/experiments/cart_pole.py --seed 0
python scripts/cip/experiments/double_pendulum.py --seed 0
python scripts/cip/experiments/triple_pendulum.py --seed 0
python scripts/cip/experiments/humulum.py --seed 0
sbatch scripts/cip/experiments/run.sh humulum        # all ten seeds as a SLURM array
```

`--objective` selects the quantity the planner maximizes: `cip` (the open-loop rate, default), `closed_loop` or `difference`. Runs of the non-default objectives go to `results/CIP_CLOSED_LOOP/` and `results/CIP_DIFFERENCE/`. In `hist.npy` the columns are (difference, CIP, closed-loop rate).

**Baselines (Table 1).**

```bash
python scripts/empowerment/run.py --task CART_POLE                # deterministic; also DOUBLE_PENDULUM, TRIPLE_PENDULUM, HUMULUM
scripts/state_entropy/experiments/run_cart_pole_seeds.sh 0 9      # also run_double_pendulum_seeds.sh, run_triple_pendulum_seeds.sh, run_humulum_seeds.sh
```

DADS, DIAYN, SMM and ICM were run with modified copies of their official implementations, pinned as submodules under `baselines/` (see `baselines/README.md`). The plotting script reads their exported height arrays from `results/<BASELINE>/<ENVIRONMENT>/`.

**Figures and tables.**

```bash
python scripts/cip/plots/plot_humulum_trajectory.py     # Figure 1
python scripts/cip/plots/plot_combined_height.py        # Figure 4 and Table 1
python scripts/cip/plots/plot_rate.py                   # Figure 5
```

**Ball in box (Appendix A.1).**

```bash
scripts/cip/experiments/run_ball_in_box_seeds.sh 0 49
scripts/random/experiments/run_ball_in_box_seeds.sh 0 49
python scripts/cip/plots/plot_ball_in_box.py            # Figure 6
```

**Lorenz and Lorenz-96 (Section 3.4, Appendix A.2).** CPU is enough.

```bash
python scripts/kse_estimation/plot_lorenz_kse.py        # Figure 3, seconds
python scripts/kse_estimation/table_lorenz96.py         # Table 2, about an hour
```

**Project webpage.** `scripts/webpage/` holds the video and freeze-frame scripts that are not part of the paper.

## Citation

```bibtex
@inproceedings{shah2026controllable,
  title={Emergence of Physical Intelligence via Controllable Information Production},
  author={Shah, Tristan and Tiomkin, Stas},
  journal={arXiv preprint arXiv:2601.22449},
  year={2026},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {39, Main Conference},
  note = {Oral presentation}
}
```
