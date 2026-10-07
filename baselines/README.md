# Learned baselines (DADS, DIAYN, SMM, ICM)

The learned intrinsic-motivation baselines of Table 1 and Figure 4 were run with modified copies of
two public codebases, included here as git submodules pinned to the exact commits used for the paper.
They never update on their own; clone with `git clone --recursive` (or run `git submodule update --init`).

| Submodule | Upstream | Baselines | Our changes |
|---|---|---|---|
| `url_benchmark/` | [URLB](https://github.com/rll-research/url_benchmark) (MIT) | DIAYN, SMM, ICM | the four environments as dm_control tasks, the hanging gibbon reset, height evaluation scripts |
| `dads/` | [DADS](https://github.com/google-research/dads) (Apache 2.0) | DADS | the four environments as gym tasks, per-environment evaluation scripts, training configs |

Each submodule has its own conda environment (`conda_env_linux.yml` / `conda_env_mac.yml`) and README.
Every baseline is pre-trained for 2M environment steps with default hyperparameters, reset deterministically
to the hanging pose. Evaluation picks the skill with the highest mean extremity height and rolls it out from
the hanging pose for several seeds, producing a `(seeds, steps)` height array that
`scripts/cip/plots/plot_combined_height.py` reads from `results/<BASELINE>/<ENVIRONMENT>/`.

## URLB: DIAYN, SMM, ICM

```bash
cd baselines/url_benchmark
python pretrain.py domain=cart_pole agent=diayn obs_type=states seed=1      # domains: cart_pole, double_pendulum, triple_pendulum, humulum; agents: diayn, smm, icm
python eval_extremity_height.py models/states/cart_pole/diayn/1/snapshot_2000010.pt --num-seeds 10
```

`eval_extremity_height.py` writes `extremity_height_<domain>_<agent>_1.npy`; copy it to
`results/<AGENT>/<ENVIRONMENT>/` under the same name (for example `results/DIAYN/CART_POLE/`).
The gibbon DIAYN and SMM curves were produced with `plot_best_skill_seeds.py` on the same snapshots,
which writes `best_skill_seeds_<agent>_1.npy`; it goes to `results/<AGENT>/HUMULUM/`.

## DADS

```bash
cd baselines/dads
python unsupervised_skill_learning/dads_off.py --logdir=logs/cart_pole --flagfile=configs/cart_pole_offpolicy.txt
python unsupervised_skill_learning/eval_cart_pole.py --logdir=logs/cart_pole --flagfile=configs/cart_pole_offpolicy.txt --out_dir=logs/cart_pole/eval --num_seeds=10
```

Likewise for `double_pendulum`, `triple_pendulum` and `humulum`. The evaluation writes
`logs/<env>/eval/best_skill_seeds.npy`; copy it to `results/DADS/<ENVIRONMENT>/best_skill_seeds.npy`.

## Notes

- Training checkpoints are not in the repository (about 5 GB). The height arrays used in the paper are the
  files under `results/<BASELINE>/`.
- The baseline MuJoCo models match `xml/` apart from added cameras. The triple pendulum baselines use the
  `RK4` integrator declared in their XML, whereas the CIP runs use `implicitfast`.
