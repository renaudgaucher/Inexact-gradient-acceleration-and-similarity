"""Production comparison plots for the MNIST / Huber / ALIE sweep.

ONE figure per (Byzantine count f, heterogeneity level), comparing all four optimization methods
(D-GD, PIGS, Acc. ExtraGrad, Acc. ExtraGrad + Prox).

Metric: suboptimality  L - L*  where L* = reference_optimum (the f=0 / NoAttack PIGS optimum),
normalised per figure by the largest initial (step-0) excess loss across the four methods -- one
shared divisor, so the methods' slightly different starting losses (notably D-GD) are rescaled
equally rather than each curve being forced to start at 1. Plotted on log-y so both convergence
(f=0) and the Byzantine-induced plateaus (f>0) are visible. Set METRIC="train_loss" for the raw
loss instead.

Display follows analysis_mnist.py:
  - NB_STEPS_DISPLAYED: total iterations shown on the main axes (None = full run),
  - ZOOM: a log-y inset over the first ZOOM_STEPS iterations (initial behaviour),
  - multi-seed mean +/- 95% CI band (via stacked_train_loss); with one seed the band is just the
    line. Seed counts are read from each results dir's config.json, so more seeds work with no edit.

Run from the repo root (after Stage 3):  python analysis_mnist_alie_prod.py
Outputs: ./plot/mnist/mnist_alie_prod/<het_tag>/f_<f>.{pdf,png}
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

from expe_mnist_alie_common import (
    HETEROGENEITY, dist_name_and_param, F_PROD,
    reference_optimum, prod_run_folder, stacked_train_loss,
)

PROD_ROOT = "./results/mnist_alie/prod"
PLOT_ROOT = "./plot/mnist/mnist_alie_prod"

# Fixed method order + consistent colour per method across every figure (for easy comparison).
METHODS = ["moDSGD", "FedProxyProx", "AccExtraGrad", "AccExtraGradProx"]
DISPLAY = {
    "moDSGD": "D-GD",
    "FedProxyProx": "PIGS",
    "AccExtraGrad": "Acc. ExtraGrad",
    "AccExtraGradProx": "Acc. ExtraGrad + Prox",
}

# ---- display parameters (tweak here) --------------------------------------------
NB_STEPS_DISPLAYED = None    # total iterations on the main axes; None = full run (e.g. 3000)
ZOOM = True                  # log-y inset over the first iterations
ZOOM_STEPS = 200             # width of that inset
METRIC = "suboptimality"     # "suboptimality" (L - L*) or "train_loss"
SUBOPT_FLOOR = 1e-10         # clip L - L* to this for log-y (f=0 curves reach ~L*)


def het_label(data_dist):
    name, param = dist_name_and_param(data_dist)
    return "i.i.d." if name == "iid" else f"Dirichlet(alpha={param})"


def mean_ci(curves):
    """(mean, err) over the seed axis; err = 95% CI half-width (0 for a single seed)."""
    mean = curves.mean(axis=0)
    n = curves.shape[0]
    err = 1.96 * curves.std(axis=0) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, err


def _draw(ax, x, series, style, dash, start=1):
    for i, (m, mean, err) in enumerate(series):
        ax.plot(x[start:], mean[start:], label=DISPLAY[m], color=style[m],
                linestyle=dash[i % len(dash)], linewidth=1.8)
        if np.any(err):
            ax.fill_between(x[start:], (mean - err)[start:], (mean + err)[start:],
                            color=style[m], alpha=0.25)


def plot_setting(series, title, ylabel, out_base, style, dash,
                 nb_steps_displayed, zoom, zoom_steps):
    """series: list of (method, mean_curve, err_curve). One figure comparing the methods."""
    t = max(len(mean) for _m, mean, _e in series)
    x = np.arange(t)
    xmax = t - 1 if nb_steps_displayed is None else min(nb_steps_displayed, t - 1)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    _draw(ax, x, series, style, dash)                      # start=1 skips the step-0 log artefact
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    if zoom:
        axins = inset_axes(ax, width="42%", height="42%", loc="upper right")
        _draw(axins, x, series, style, dash)
        axins.set_yscale("log")
        axins.set_xlim(0, min(zoom_steps, xmax))
        axins.set_title("first iterations", fontsize=8)
        axins.tick_params(labelsize=8)
        axins.grid(True, which="both", alpha=0.3)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, bbox_inches="tight")  # inset-safe (no tight_layout)
    plt.close(fig)


def make_plots(nb_steps_displayed=NB_STEPS_DISPLAYED, zoom=ZOOM, zoom_steps=ZOOM_STEPS,
               metric=METRIC, subopt_floor=SUBOPT_FLOOR):
    matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
    palette = sns.color_palette("colorblind", n_colors=len(METHODS))
    style = {m: palette[i] for i, m in enumerate(METHODS)}
    dash = ["-", "--", "-.", ":"]
    ylabel = ("(L - L*) / (L0 - L*)_max  (log scale)" if metric == "suboptimality"
              else "Train loss (log scale)")

    made = 0
    for data_dist, tag in HETEROGENEITY:
        results_dir = f"{PROD_ROOT}/{tag}"
        if not os.path.isdir(results_dir):
            continue
        bc = json.load(open(os.path.join(results_dir, "config.json")))["benchmark_config"]
        n_tr, n_dd = bc["nb_training_seeds"], bc["nb_data_distribution_seeds"]
        s0_tr, s0_dd = bc["training_seed"], bc["data_distribution_seed"]
        lstar = reference_optimum(results_dir, data_dist) if metric == "suboptimality" else None

        for f in F_PROD:
            # First pass: collect the (already L*-subtracted) excess-loss curves per method.
            collected = []
            for m in METHODS:
                folder = prod_run_folder(results_dir, m, data_dist, f)
                if folder is None:
                    continue
                curves = stacked_train_loss(folder, n_tr, n_dd, s0_tr, s0_dd)
                if curves is None:
                    continue
                if metric == "suboptimality":
                    curves = curves - (lstar or 0.0)
                collected.append((m, curves))
            if not collected:
                continue

            # Fair comparison: divide every method's excess loss by ONE shared scalar -- the
            # largest initial (step-0) excess loss across methods. The methods' step-0 losses
            # differ slightly (notably D-GD vs the rest), so normalising each curve by its own
            # start would hide that gap; a single shared divisor rescales them all equally.
            denom = 1.0
            if metric == "suboptimality":
                denom = max(float(np.mean(curves[:, 0])) for _m, curves in collected)

            series = []
            for m, curves in collected:
                if metric == "suboptimality":
                    curves = np.maximum(curves / denom, subopt_floor)
                mean, err = mean_ci(curves)
                series.append((m, mean, err))

            title = f"{het_label(data_dist)},  f = {f} Byzantine  (40 honest, ALIE)"
            out_base = os.path.join(PLOT_ROOT, tag, f"f_{f}")
            plot_setting(series, title, ylabel, out_base, style, dash,
                         nb_steps_displayed, zoom, zoom_steps)
            made += 1
            print(f"[{tag}] f={f}: {len(series)} methods"
                  f"{'' if lstar is None else f', L*={lstar:.4f}, (L0-L*)_max={denom:.4f}'}"
                  f"  ->  {out_base}.pdf")

    print(f"\nDone: {made} figures under {PLOT_ROOT}/")


if __name__ == "__main__":
    make_plots()
