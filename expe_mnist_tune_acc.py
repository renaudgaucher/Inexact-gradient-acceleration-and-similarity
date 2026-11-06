"""Stage 2 - tune PIGS / AccExtraGrad(+prox) at f=0 (NB_STEPS_TUNE steps).

Loops over the 4 heterogeneity levels. For each, in TWO passes (the prox grid depends on the
no-prox result):
  Pass 1: PIGS and Acc. ExtraGrad (no prox), both anchored on eta*_GD from Stage 1
    - PIGS           = pigs_multipliers(het) * eta*_GD   (extended grid for iid / Dirichlet alpha>=2)
    - Acc. ExtraGrad = EXTRAGRAD_MULT * eta*_GD
  Pass 2: Acc. ExtraGrad + prox, anchored on the NO-PROX tuned LR eta*_aeg
    - = EXTRAGRADPROX_MULT * eta*_aeg   (i.e. {1,2,5,10,50,100} * eta*_aeg)
Both passes write to the same per-het results dir. Each het is a separate config / results subdir.

Requires Stage 1.  Run from the repo root:  python expe_mnist_tune_acc.py
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from byzfl import run_benchmark

from expe_mnist_alie_common import (
    base_config, pigs, accextragrad, accextragradprox,
    select_best_lr_by_train_loss, anchored_grid, pigs_multipliers,
    EXTRAGRAD_MULT, EXTRAGRADPROX_MULT, NB_STEPS_TUNE, HETEROGENEITY,
)

GD_DIR = "./results/mnist_alie/tune_gd"
OUT_ROOT = "./results/mnist_alie/tune_acc"
NB_JOBS = 8

# Manual eta*_GD overrides: het tag -> the GD anchor LR used to build the PIGS / AccExtraGrad grids.
# Mirrors LR_OVERRIDES in expe_mnist_prod.py. The Stage-1 selector can pick a fast-but-unstable
# eta*_GD (edge of stability), which then anchors the whole Stage-2 grid on a bad value. Pin a value
# here to force the anchor; any tag NOT listed falls back to select_best_lr_by_train_loss on GD_DIR.
# Currently forced to 0.1 for every het (treat the optimal GD LR as 0.1).
GD_LR_OVERRIDES = {
    "iid":    0.1,
    "dir0.1": 0.1,
    "dir1.0": 0.1,
    "dir5.0": 0.1,
}


def pick_eta_gd(tag, data_dist):
    """eta*_GD from GD_LR_OVERRIDES[tag] if set, else the automatic area selector on GD_DIR."""
    if tag in GD_LR_OVERRIDES:
        print(f"[{tag}] moDSGD: using manual override eta*_GD={GD_LR_OVERRIDES[tag]}")
        return GD_LR_OVERRIDES[tag]
    return select_best_lr_by_train_loss(GD_DIR, "moDSGD", data_dist, nb_seeds=1)


def run_config(results_dir, data_dist, methods):
    config = base_config(
        results_directory=results_dir,
        data_distribution=[data_dist],
        training_algorithms=methods,
        f_list=[0],
        nb_training_seeds=1,
        nb_steps=NB_STEPS_TUNE,
    )
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    run_benchmark(NB_JOBS)


def main():
    for data_dist, tag in HETEROGENEITY:
        results_dir = f"{OUT_ROOT}/{tag}"
        eta_gd = pick_eta_gd(tag, data_dist)

        # Pass 1: PIGS + Acc. ExtraGrad (no prox), both anchored on eta*_GD.
        pigs_grid = anchored_grid(pigs_multipliers(data_dist), eta_gd)
        aeg_grid = anchored_grid(EXTRAGRAD_MULT, eta_gd)
        print(f"[{tag}] eta*_GD={eta_gd}  ->  PIGS {pigs_grid}, AccExtraGrad {aeg_grid}")
        run_config(results_dir, data_dist, [pigs(pigs_grid), accextragrad(aeg_grid)])

        # Pass 2: Acc. ExtraGrad + prox, anchored on the NO-PROX tuned LR eta*_aeg.
        eta_aeg = select_best_lr_by_train_loss(results_dir, "AccExtraGrad", data_dist, nb_seeds=1)
        aegp_grid = anchored_grid(EXTRAGRADPROX_MULT, eta_aeg)
        print(f"[{tag}] eta*_AccExtraGrad={eta_aeg}  ->  AccExtraGradProx {aegp_grid}")
        run_config(results_dir, data_dist, [accextragradprox(aegp_grid)])


if __name__ == "__main__":
    main()
