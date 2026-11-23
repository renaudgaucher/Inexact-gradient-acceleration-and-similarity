"""Stage 3 - production sweep: all four methods @ tuned LR across the Byzantine f sweep.

Loops over the 4 heterogeneity levels. For each:
  - a REFERENCE run: PIGS at f=0 with NoAttack -> achievable optimum L* (same objective as the
    sweep: same weight_decay / aggregator / horizon), so L* is the true floor of the production
    curves. Used to plot suboptimality L - L* (read it back with `reference_optimum`).
  - the main sweep: all four methods at their tuned LR across f = {0,1,5,10,20} (40 honest + f
    Byzantine) under ALIE.
The reference runs FIRST so the sweep's config.json is the one left on disk (plot_metric_curve reads
it). Both run folders coexist in prod/<tag>. One config / results subdir per het.

Requires Stages 1-2.  Run from the repo root:  python expe_mnist_prod.py
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from byzfl import run_benchmark

from expe_mnist_alie_common import (
    base_config, gd, pigs, accextragrad, accextragradprox,
    select_best_lr_by_train_loss, ATTACK_NAME, REF_ATTACK, F_PROD, HETEROGENEITY,
)

GD_DIR = "./results/mnist_alie/tune_gd"
ACC_ROOT = "./results/mnist_alie/tune_acc"
OUT_ROOT = "./results/mnist_alie/prod"
NB_SEEDS = 1
NB_JOBS = 8

# Manual LR overrides: (het tag, method config-name) -> learning rate.
# The automatic selector picks the lowest-AREA LR, which can prefer a fast-but-unstable trajectory
# (e.g. Dirichlet 0.1). Inspect each method with `print_lr_diagnostics` (see inspect_lrs.py) and, when
# the area optimum is unstable, pin a stable LR here. Any (tag, method) NOT listed uses the selector.
LR_OVERRIDES = {
    ("dir0.1", "moDSGD"): 0.1,
    ("dir0.1", "AccExtraGrad"): 0.1,
    ("dir0.1", "FedProxyProx"): 0.5,
    ("dir0.1", "AccExtraGradProx"): 0.5,
}


def pick_lr(auto_dir, algo, data_dist, tag):
    """LR from LR_OVERRIDES[(tag, algo)] if set, else the automatic area selector on `auto_dir`."""
    key = (tag, algo)
    if key in LR_OVERRIDES:
        print(f"[{tag}] {algo}: using manual override lr={LR_OVERRIDES[key]}")
        return LR_OVERRIDES[key]
    return select_best_lr_by_train_loss(auto_dir, algo, data_dist)


def run_config(results_dir, data_dist, methods, f_list, nb_seeds, attack_name=ATTACK_NAME):
    config = base_config(
        results_directory=results_dir,
        data_distribution=[data_dist],
        training_algorithms=methods,
        f_list=f_list,
        nb_training_seeds=nb_seeds,
        attack_name=attack_name,
    )
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    run_benchmark(NB_JOBS)


def main():
    for data_dist, tag in HETEROGENEITY:
        results_dir = f"{OUT_ROOT}/{tag}"
        acc_dir = f"{ACC_ROOT}/{tag}"
        lr_gd   = pick_lr(GD_DIR,  "moDSGD",           data_dist, tag)
        lr_pigs = pick_lr(acc_dir, "FedProxyProx",     data_dist, tag)
        lr_aeg  = pick_lr(acc_dir, "AccExtraGrad",     data_dist, tag)
        lr_aegp = pick_lr(acc_dir, "AccExtraGradProx", data_dist, tag)
        print(f"[{tag}] tuned LRs -> GD={lr_gd}  PIGS={lr_pigs}  "
              f"AccExtraGrad={lr_aeg}  AccExtraGradProx={lr_aegp}")

        # Reference run FIRST (PIGS, f=0, NoAttack) -> achievable optimum L*. Deterministic
        # (full batch, f=0, fixed seed), so a single seed suffices.
        run_config(results_dir, data_dist, [pigs([lr_pigs])], f_list=[0], nb_seeds=1,
                   attack_name=REF_ATTACK)

        # Main sweep: all four methods across the Byzantine f sweep under ALIE.
        methods = [gd([lr_gd]), pigs([lr_pigs]), accextragradprox([lr_aegp]), accextragrad([lr_aeg])]
        run_config(results_dir, data_dist, methods, f_list=F_PROD, nb_seeds=NB_SEEDS)


if __name__ == "__main__":
    main()
