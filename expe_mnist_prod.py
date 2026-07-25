"""Stage 3 - production sweep: all four methods @ tuned LR across the Byzantine f sweep.

Loops over the 4 heterogeneity levels. For each: pick every method's tuned LR (lowest train loss)
from Stages 1-2, then run f = {0,1,5,10,20} (40 honest + f Byzantine) with 5 training seeds.
One config / results subdir per het.

Requires Stages 1-2 to have run.  Run from the repo root:  python expe_mnist_prod.py
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from byzfl import run_benchmark

from expe_mnist_alie_common import (
    base_config, gd, pigs, accextragrad, accextragradprox,
    select_best_lr_by_train_loss, F_PROD, HETEROGENEITY,
)

GD_DIR = "./results/mnist_alie/tune_gd"
ACC_ROOT = "./results/mnist_alie/tune_acc"
OUT_ROOT = "./results/mnist_alie/prod"
NB_SEEDS = 1
NB_JOBS = 10


def main():
    for data_dist, tag in HETEROGENEITY:
        acc_dir = f"{ACC_ROOT}/{tag}"
        lr_gd   = select_best_lr_by_train_loss(GD_DIR,  "moDSGD",           data_dist)
        lr_pigs = select_best_lr_by_train_loss(acc_dir, "FedProxyProx",     data_dist)
        lr_aeg  = select_best_lr_by_train_loss(acc_dir, "AccExtraGrad",     data_dist)
        lr_aegp = select_best_lr_by_train_loss(acc_dir, "AccExtraGradProx", data_dist)
        print(f"[{tag}] tuned LRs -> GD={lr_gd}  PIGS={lr_pigs}  "
              f"AccExtraGrad={lr_aeg}  AccExtraGradProx={lr_aegp}")

        methods = [
            gd([lr_gd]),
            pigs([lr_pigs]),
            accextragradprox([lr_aegp]),
            accextragrad([lr_aeg]),
        ]
        config = base_config(
            results_directory=f"{OUT_ROOT}/{tag}",
            data_distribution=[data_dist],
            training_algorithms=methods,
            f_list=F_PROD,
            nb_training_seeds=NB_SEEDS,
        )
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        run_benchmark(NB_JOBS)


if __name__ == "__main__":
    main()
