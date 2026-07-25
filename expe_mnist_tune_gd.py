"""Stage 1 - tune GD (moDSGD) at f=0, per heterogeneity level.

One config: moDSGD only, f=[0], all 4 heterogeneity levels in a single data_distribution list,
learning_rate = GD_GRID, 1 seed. The winning LR per het (lowest train loss) is read on demand by
Stage 2 via `select_best_lr_by_train_loss` -- no best-LR file is produced here.

Run from the repo root:  python expe_mnist_tune_gd.py
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from byzfl import run_benchmark

from expe_mnist_alie_common import base_config, gd, GD_GRID, HETEROGENEITY

RESULTS_DIR = "./results/mnist_alie/tune_gd"
NB_JOBS = 8


def main():
    data_distribution = [entry for entry, _tag in HETEROGENEITY]
    config = base_config(
        results_directory=RESULTS_DIR,
        data_distribution=data_distribution,
        training_algorithms=[gd(GD_GRID)],
        f_list=[0],
        nb_training_seeds=1,
    )
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    run_benchmark(NB_JOBS)


if __name__ == "__main__":
    main()
