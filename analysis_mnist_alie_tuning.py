"""Plot the LR-tuning results (Stages 1-2) of the MNIST / Huber / ALIE sweep.

One figure per (heterogeneity level x algorithm), each showing the train-loss curve of EVERY
candidate learning rate, with the selected (lowest train loss) LR starred. This is the diagnostic
that shows why each per-method LR was chosen.

Note: the built-in `plot_metric_curve` cannot do this -- `for_each_experiment` treats the LR grid as
one atomic scenario attribute, so it only ever draws a single LR per algorithm. We read the per-LR
run folders directly instead (via `train_loss_curves_by_lr`).

Run from the repo root (after Stages 1-2):  python analysis_mnist_alie_tuning.py
Outputs: ./plot/mnist/mnist_alie_tuning/<het_tag>/<algo>.{pdf,png}
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from expe_mnist_alie_common import (
    HETEROGENEITY, dist_name_and_param, train_loss_curves_by_lr,
)

GD_DIR = "./results/mnist_alie/tune_gd"
ACC_ROOT = "./results/mnist_alie/tune_acc"
PLOT_ROOT = "./plot/mnist/mnist_alie_tuning"
NB_SEEDS = 1  # tuning stages use a single seed

# moDSGD lives in the shared tune_gd dir; the robust methods live in per-het tune_acc/<tag> dirs.
ALGORITHMS = ["moDSGD", "FedProxyProx", "AccExtraGrad", "AccExtraGradProx"]
ALGO_DISPLAY = {
    "moDSGD": "D-GD",
    "FedProxyProx": "PIGS",
    "AccExtraGrad": "Acc. ExtraGrad (no prox)",
    "AccExtraGradProx": "Acc. ExtraGrad + Prox",
}


def algo_results_dir(algo, tag):
    return GD_DIR if algo == "moDSGD" else f"{ACC_ROOT}/{tag}"


def het_label(data_dist):
    name, param = dist_name_and_param(data_dist)
    return "i.i.d." if name == "iid" else f"Dirichlet(alpha={param})"


def plot_lr_grid(curves, title, out_base):
    """curves: list of (lr, train_loss_curve). Draw all LRs on one log-y figure; star the best."""
    # Best = lowest area under the curve (mean over steps), matching select_best_lr_by_train_loss.
    finite = [(lr, c) for lr, c in curves if np.isfinite(np.mean(c))]
    best_lr = min(finite, key=lambda kv: np.mean(kv[1]))[0] if finite else None

    palette = sns.color_palette("colorblind", n_colors=max(len(curves), 3))
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (lr, curve) in enumerate(curves):           # already sorted by lr
        is_best = (lr == best_lr)
        ax.plot(
            np.arange(1,1+len(curve)), curve,        # x starts at 1 for log-x
            color=palette[i],
            alpha=0.6,
            label=f"lr={lr}" + ("  ★ best" if is_best else ""),
        )
    # log-log: these full-batch problems converge within ~1e2 steps, so log-x spreads out the
    # transient (where the LRs actually differ) instead of squashing it against a long flat tail.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="learning rate", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)
    return best_lr


def main():
    made, skipped = 0, []
    for data_dist, tag in HETEROGENEITY:
        for algo in ALGORITHMS:
            results_dir = algo_results_dir(algo, tag)
            curves = train_loss_curves_by_lr(results_dir, algo, data_dist, nb_seeds=NB_SEEDS)
            if not curves:
                skipped.append((tag, algo))
                continue
            title = f"{ALGO_DISPLAY[algo]}  |  {het_label(data_dist)}  (f=0)"
            out_base = os.path.join(PLOT_ROOT, tag, algo)
            best = plot_lr_grid(curves, title, out_base)
            made += 1
            print(f"[{tag}] {algo}: {len(curves)} LRs, best={best}  ->  {out_base}.pdf")

    print(f"\nDone: {made} figures under {PLOT_ROOT}/")
    if skipped:
        print("Skipped (no results found):", skipped)


if __name__ == "__main__":
    main()
