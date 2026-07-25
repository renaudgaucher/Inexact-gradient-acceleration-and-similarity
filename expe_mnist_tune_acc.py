"""Stage 2 - tune PIGS / AccExtraGrad(+prox) at f=0, LR grids anchored on GD's tuned LR.

Loops over the 4 heterogeneity levels. For each: read eta*_GD(het) from Stage 1 (lowest train loss),
then sweep the three robust methods with grids anchored on it:
  - PIGS = {5,10,20,50}*eta*  (iid adds a larger point: {..,100}*eta*),
  - Acc. ExtraGrad (no prox)  = {0.1,0.2,0.5,1,2}*eta*,
  - Acc. ExtraGrad + prox     = {0.2,0.5,1,2,5}*eta*  (tolerates larger steps than the plain one).
Because the grids differ per het, each het is a separate config / results subdir.

Requires Stage 1 to have run.  Run from the repo root:  python expe_mnist_tune_acc.py
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from byzfl import run_benchmark

from expe_mnist_alie_common import (
    base_config, pigs, accextragrad, accextragradprox,
    select_best_lr_by_train_loss, anchored_grid,
    PIGS_MULT, PIGS_MULT_IID, EXTRAGRAD_MULT, EXTRAGRADPROX_MULT, HETEROGENEITY,
)

GD_DIR = "./results/mnist_alie/tune_gd"
OUT_ROOT = "./results/mnist_alie/tune_acc"
NB_JOBS = 8


def main():
    for data_dist, tag in HETEROGENEITY:
        eta_gd = select_best_lr_by_train_loss(GD_DIR, "moDSGD", data_dist, nb_seeds=1)

        # iid (very homogeneous) lets PIGS push to a larger step-size, so it gets the extended grid.
        pigs_mult = PIGS_MULT_IID if tag == "iid" else PIGS_MULT
        pigs_grid = anchored_grid(pigs_mult, eta_gd)
        aeg_grid = anchored_grid(EXTRAGRAD_MULT, eta_gd)           # no prox
        aegp_grid = anchored_grid(EXTRAGRADPROX_MULT, eta_gd)      # with prox (larger steps allowed)
        print(f"[{tag}] eta*_GD = {eta_gd}  ->  PIGS {pigs_grid}, "
              f"AccExtraGrad {aeg_grid}, AccExtraGradProx {aegp_grid}")

        methods = [
            pigs(pigs_grid),
            accextragradprox(aegp_grid),
            accextragrad(aeg_grid),
        ]
        config = base_config(
            results_directory=f"{OUT_ROOT}/{tag}",
            data_distribution=[data_dist],
            training_algorithms=methods,
            f_list=[0],
            nb_training_seeds=1,
        )
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        run_benchmark(NB_JOBS)


if __name__ == "__main__":
    main()
