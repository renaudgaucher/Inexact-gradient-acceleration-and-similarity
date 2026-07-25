"""Shared config + LR-selection helpers for the MNIST / Huber / ALIE staged sweep.

Three chained scripts use this module:
  1. expe_mnist_tune_gd.py   -- tune GD (moDSGD) at f=0, per heterogeneity level.
  2. expe_mnist_tune_acc.py  -- tune PIGS / AccExtraGrad(+prox) at f=0, LR grids anchored on GD's.
  3. expe_mnist_prod.py      -- production sweep, all methods @ tuned LR across the f sweep.

We care only about the optimization metric (training loss), so learning rates are selected by
LOWEST train loss (see `select_best_lr_by_train_loss`), not validation accuracy -- i.e. we do NOT
use the library's accuracy-based `find_best_hyperparameters`.

Fixed method hyper-parameters mirror expe_mnist.py exactly; only the learning rates are tuned.
"""

import os
import glob

import numpy as np

# ---- fixed experiment constants -------------------------------------------------
NB_HONEST = 40                      # honest clients (Byzantine added on top: n = 40 + f)
F_PROD = [0, 1, 5, 10, 20]          # Byzantine counts for the production sweep
NB_STEPS = 5000
EVAL_DELTA = 100
WEIGHT_DECAY = 0.01
AGG_NAME = "Huber"
ATTACK_NAME = "Optimal_ALittleIsEnough"   # "ALIE" (no-op at f=0)

# ---- learning-rate search grids -------------------------------------------------
# GD is an absolute grid; the other methods are anchored on GD's tuned LR eta* (see anchored_grid):
# PIGS larger than GD, extra-gradient around/below GD (per the accelerated-method behaviour).
GD_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
PIGS_MULT = [5, 10, 20, 50]
EXTRAGRAD_MULT = [0.2, 0.5, 1.0, 2.0, 5.0]

# ---- heterogeneity levels: (data_distribution entry, folder tag) ----------------
# iid + Dirichlet(alpha) for alpha in {0.1, 1, 5}. Floats (1.0/5.0) and None (iid) match the
# run-folder naming produced by FileManager / train.py.
HETEROGENEITY = [
    ({"name": "iid",            "distribution_parameter": [None]}, "iid"),
    ({"name": "dirichlet_niid", "distribution_parameter": [0.1]},  "dir0.1"),
    ({"name": "dirichlet_niid", "distribution_parameter": [1.0]},  "dir1.0"),
    ({"name": "dirichlet_niid", "distribution_parameter": [5.0]},  "dir5.0"),
]


def dist_name_and_param(data_dist):
    """(name, scalar distribution_parameter) exactly as they appear in run-folder names."""
    name = data_dist["name"]
    # iid / extreme_niid force the parameter to None in the folder name (see train.py:93-97).
    param = None if name in ("iid", "extreme_niid") else data_dist["distribution_parameter"][0]
    return name, param


def anchored_grid(mults, eta):
    """LR grid for a method anchored on GD's tuned LR `eta` (rounded to kill float dust)."""
    return [round(m * eta, 10) for m in mults]


# ---- per-method config builders (fixed HPs copied from expe_mnist.py lines 30-79) ----
def gd(lr_list):
    return {
        "name": "moDSGD",  # plain GD
        "parameters": {
            "optimizer_name": "SGD",
            "momentum": [0.0],
            "optimizer_parameters": {},
            "learning_rate": list(lr_list),
        },
    }


def pigs(lr_list):
    return {
        "name": "FedProxyProx",  # PIGS
        "parameters": {
            "optimizer_name": "SGD",
            "momentum": [0.0],
            "optimizer_parameters": {},
            "learning_rate": list(lr_list),
            "prox_optimizer_name": "SGD",
            "prox_optimizer_params": {"learning_rate": 0.05, "security_factor": 10},
        },
    }


def accextragradprox(lr_list):
    return {
        "name": "AccExtraGradProx",
        "parameters": {
            "optimizer_name": "SGD",
            "optimizer_parameters": {},
            "learning_rate": list(lr_list),
            "momentum": [WEIGHT_DECAY],  # gamma (strong-convexity proxy), NOT classical momentum; < 1/lr
            "tau_factor": 1,
            "fast_lr_factor": 1,
            "prox_optimizer_name": "SGD",
            "prox_optimizer_params": {"learning_rate": 0.05, "security_factor": 10},
        },
    }


def accextragrad(lr_list):
    return {
        "name": "AccExtraGrad",
        "parameters": {
            "optimizer_name": "SGD",
            "optimizer_parameters": {},
            "learning_rate": list(lr_list),
            "momentum": [WEIGHT_DECAY],  # gamma, as above
            "tau_factor": 1,
            "fast_lr_factor": 1.0,
        },
    }


def base_config(results_directory, data_distribution, training_algorithms,
                f_list, nb_training_seeds=1, nb_steps=NB_STEPS):
    """Full benchmark config dict. `data_distribution` is a list of heterogeneity entries.

    size_train_set stays 0.8: the eval loop raises without a val split (train.py:389), so 1.0 is
    unusable. Val accuracy is computed/written but ignored; train loss is on the 80% train split.
    """
    return {
        "benchmark_config": {
            "dtype": "float64",
            "device": "cpu",
            "training_seed": 0,
            "nb_training_seeds": nb_training_seeds,
            "nb_honest_clients": NB_HONEST,
            "f": list(f_list),
            "size_train_set": 0.8,
            "data_distribution_seed": 0,
            "nb_data_distribution_seeds": 1,
            "data_distribution": data_distribution,
            "training_algorithm": training_algorithms,
            "nb_steps": nb_steps,
        },
        "model": {
            "name": "logreg_mnist",
            "dataset_name": "mnist",
            "nb_labels": 10,
            "loss": "NLLLoss",
            "weight_decay": [WEIGHT_DECAY],
        },
        "aggregator": [{"name": AGG_NAME, "parameters": {}}],
        "pre_aggregators": [],
        "honest_clients": {"batch_size": 0},  # 0 = full batch (GD)
        "attack": [{"name": ATTACK_NAME, "parameters": {}}],
        "evaluation_and_results": {
            "evaluation_delta": EVAL_DELTA,
            "batch_size_evaluation": 2 ** 6,
            "evaluate_on_test": False,          # optimization metric only -> skip test accuracy
            "store_per_client_metrics": False,
            "store_models": False,
            "data_folder": "./data",
            "results_directory": results_directory,
        },
    }


def select_best_lr_by_train_loss(results_dir, algo, data_dist, nb_seeds=1):
    """Return the LR (float) with the lowest train loss for `algo` at n=40, f=0, this het.

    Globs the per-LR run folders for the method/heterogeneity, reads each train-loss curve, averages
    over seeds and takes the min over steps. Glob-based, so it is robust to LR float formatting and
    needs no knowledge of the grid that produced the runs. Non-finite (diverged) curves are ignored.
    """
    # Lazy import: keeps this module light for spawned workers (evaluate_results pulls matplotlib).
    from byzfl.benchmark.evaluate_results import experiment_base_name

    dist_name, param = dist_name_and_param(data_dist)
    base = experiment_base_name("mnist", "logreg_mnist", algo,
                                NB_HONEST, 0, 0, dist_name, param, AGG_NAME, "")
    # The "_n_" boundary after the algo name keeps this prefix from matching a longer algo name
    # (e.g. AccExtraGrad vs AccExtraGradProx).
    prefix = f"{base}_{ATTACK_NAME}_lr_"

    best_lr, best_val = None, np.inf
    for folder in glob.glob(os.path.join(results_dir, prefix + "*")):
        lr_str = os.path.basename(folder)[len(prefix):].split("_mom_")[0]
        curves = []
        for s in range(nb_seeds):
            fp = os.path.join(folder, f"train_loss_tr_seed_{s}_dd_seed_0.txt")
            if os.path.exists(fp):
                curves.append(np.genfromtxt(fp, delimiter=","))
        if not curves:
            continue
        val = float(np.min(np.mean(curves, axis=0)))   # min over steps of seed-mean train loss
        if np.isfinite(val) and val < best_val:
            best_lr, best_val = float(lr_str), val

    if best_lr is None:
        raise RuntimeError(
            f"No usable train-loss results for algo={algo!r} in {results_dir!r} "
            f"(prefix {prefix!r}). Did the previous tuning stage run to completion?"
        )
    return best_lr
