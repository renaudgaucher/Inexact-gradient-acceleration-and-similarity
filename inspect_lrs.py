"""Inspect the tuned learning rates before locking prod LRs.

For every (heterogeneity level, method) it prints a per-LR table:
  area   -- mean over all steps (what select_best_lr_by_train_loss minimises),
  tail   -- mean over the last 50% of steps (the floor it settles to),
  wobble -- mean |dL|/L over that tail (step-to-step jitter; large == oscillating / unstable),
and flags the area optimum (<-area) and the stability-filtered pick (<-STABLE-BEST).

Use it to spot the "fast but unstable" case (low area, high wobble) -- e.g. Dirichlet 0.1 -- then
pin a stable LR in expe_mnist_prod.py's LR_OVERRIDES.

Run from the repo root (after Stages 1-2):  python inspect_lrs.py
"""

from expe_mnist_alie_common import HETEROGENEITY, print_lr_diagnostics

GD_DIR = "./results/mnist_alie/tune_gd"
ACC_ROOT = "./results/mnist_alie/tune_acc"

# (auto results dir, method config-name) for each method.
TARGETS = [
    (GD_DIR, "moDSGD"),
    ("{acc}", "FedProxyProx"),
    ("{acc}", "AccExtraGrad"),
    ("{acc}", "AccExtraGradProx"),
]

# Tail window and instability bar; raise max_wobble to be more permissive.
TAIL_FRAC = 0.5
MAX_WOBBLE = 0.02


def main():
    for data_dist, tag in HETEROGENEITY:
        print(f"\n================ {tag} ================")
        acc_dir = f"{ACC_ROOT}/{tag}"
        for dir_tmpl, algo in TARGETS:
            results_dir = dir_tmpl.format(acc=acc_dir)
            print_lr_diagnostics(results_dir, algo, data_dist,
                                 tail_frac=TAIL_FRAC, max_wobble=MAX_WOBBLE)


if __name__ == "__main__":
    main()
