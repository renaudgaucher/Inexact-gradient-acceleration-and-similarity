"""Tabular summary of the MNIST / Huber / ALIE production sweep.

Each cell = (asymptotic error, #iterations to reach `factor` x asymptotic error) for ONE run,
identified by the triple (algorithm, heterogeneity, Byzantine count f).

  - "asymptotic error" = the plateau the curve settles to, estimated as the mean over the last
    TAIL_FRAC of the (seed-averaged) curve. Metric is suboptimality  L - L*  by default
    (L* = reference_optimum, the f=0 / NoAttack PIGS optimum); use --metric train_loss for raw loss.
  - "#iterations"       = the first iteration that reaches the plateau to within (factor-1), i.e.
    crosses the near edge of the +/-(factor-1) band around L_inf (factor default 1.05, i.e. 5%). The
    direction follows L_inf vs the start L_0: "first time below" when the error decays into the
    plateau, "first time above" when the Byzantine error grows into it (see asymptotic_stats). Train
    loss is logged once per step, so the array index IS the iteration number.

Three varying parameters -- f, heterogeneity, algorithm -- and you pick which TWO become the table's
rows / columns; the third is sliced (ONE table printed per value of it).

Run from the repo root (after Stage 3), e.g.:
  python analysis_mnist_alie_table.py --rows algo --cols f            # one table per heterogeneity
  python analysis_mnist_alie_table.py --rows het  --cols algo         # one table per f
  python analysis_mnist_alie_table.py --rows f    --cols het --metric train_loss
Outputs: printed tables + CSVs under ./tables/mnist_alie_prod/ (numeric long form + per-slice cells).
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from expe_mnist_alie_common import (
    HETEROGENEITY, F_PROD,
    reference_optimum, prod_run_folder, stacked_train_loss,
)

PROD_ROOT = "./results/mnist_alie/prod"
TABLE_ROOT = "./tables/mnist_alie_prod"

METHODS = ["moDSGD", "FedProxyProx", "AccExtraGrad", "AccExtraGradProx"]
DISPLAY = {
    "moDSGD": "D-GD",
    "FedProxyProx": "PIGS",
    "AccExtraGrad": "Acc. ExtraGrad",
    "AccExtraGradProx": "Acc. ExtraGrad + Prox",
}

DIMS = ("algo", "het", "f")           # the three varying parameters
COL = {"algo": "algo_label", "het": "het", "f": "f"}   # dataframe column backing each dim
DIM_TITLE = {"algo": "algorithm", "het": "heterogeneity", "f": "f (Byzantine)"}

DEFAULT_METRIC = "suboptimality"      # or "train_loss"
DEFAULT_TAIL_FRAC = 0.1               # fraction of the tail used to estimate the plateau
DEFAULT_FACTOR = 1.05                 # reach the plateau to within (factor-1) = 5%
F_DISPLAY = F_PROD # Remove the f=0 from the displayed number of Byzantine 
F_0_METRIC = 1e-4

# ---- per-run statistics ---------------------------------------------------------
def asymptotic_stats(curve, tail_frac, factor):
    """(asymptotic_error, iters) for a 1-D (seed-mean) metric curve.

    asymptotic_error (L_inf) = mean over the last `tail_frac` of the curve (the plateau).
    iters = first index that CROSSES the near edge of the +/-(factor-1) band around L_inf, with the
            direction set by L_inf vs the start L_0 = curve[0]:
              - L_inf < L_0 (error decays into the plateau from above): first index with
                value <= L_inf + band   -- "first time it goes below" the upper edge.
              - L_inf > L_0 (Byzantine error grows into the plateau from below): first index with
                value >= L_inf - band   -- "first time it goes above" the lower edge.
            band = |factor - 1| * |L_inf|. Only the NEAR edge is tested (a single crossing), so a
            fast method that overshoots clean through the band in one step is still credited at the
            step it crosses -- the previous "stay inside the band" test missed exactly that case.
    """
    c = np.asarray(curve, dtype=float)
    tail = c[int((1.0 - tail_frac) * len(c)):]
    err = float(np.mean(tail))
    band = abs(factor - 1.0) * abs(err)
    if err < c[0]:
        hit = np.where(c <= err + band)[0]   # decay from above -> first time it goes below
    else:
        hit = np.where(c >= err - band)[0]   # growth from below -> first time it goes above
    iters = int(hit[0]) if hit.size else None
    return err, iters

def finite_error_stats(curve, threshold_err):
    c = np.asarray(curve, dtype=float)
    err = threshold_err
    if err < c[0]:
        hit = np.where(c <= err)[0]   # decay from above -> first time it goes below
    else:
        hit = np.where(c >= err)[0]   # growth from below -> first time it goes above
    iters = int(hit[0]) if hit.size else None
    return err, iters

def build_records(metric, tail_frac, factor, prod_root=PROD_ROOT):
    """Long-form dataframe: one row per (algo, het, f) run with its (err, iters)."""
    records = []
    for data_dist, tag in HETEROGENEITY:
        results_dir = f"{prod_root}/{tag}"
        if not os.path.isdir(results_dir):
            continue
        bc = json.load(open(os.path.join(results_dir, "config.json")))["benchmark_config"]
        n_tr, n_dd = bc["nb_training_seeds"], bc["nb_data_distribution_seeds"]
        s0_tr, s0_dd = bc["training_seed"], bc["data_distribution_seed"]
        lstar = reference_optimum(results_dir, data_dist) if metric == "suboptimality" else None

        for algo in METHODS:
            for f in F_DISPLAY:
                folder = prod_run_folder(results_dir, algo, data_dist, f)
                if folder is None:
                    continue
                curves = stacked_train_loss(folder, n_tr, n_dd, s0_tr, s0_dd)
                if curves is None:
                    continue
                curve = curves.mean(axis=0)
                if metric == "suboptimality":
                    curve = curve - (lstar or 0.0)
                if f > 0:
                    err, iters = asymptotic_stats(curve, tail_frac, factor)
                else:
                    err, iters = finite_error_stats(curve, F_0_METRIC)
                
                records.append({
                    "algo": algo, "algo_label": DISPLAY[algo],
                    "het": tag, "f": f, "err": err, "iters": iters,
                })
    return pd.DataFrame(records)


# ---- pivoting into (rows x cols) tables, sliced over the third dim ---------------
def _dim_order(dim, present):
    """Canonical order for a dim's values, restricted to those actually present."""
    if dim == "algo":
        return [DISPLAY[m] for m in METHODS if DISPLAY[m] in present]
    if dim == "het":
        return [tag for _dd, tag in HETEROGENEITY if tag in present]
    if dim == "f":
        return [x for x in F_DISPLAY if x in present]
    return sorted(present)


def _fmt_cell(err, iters):
    if err is None or (isinstance(err, float) and np.isnan(err)):
        return ""
    it = "--" if iters is None else str(int(iters))
    return f"{err:.1e} | {it}"


def make_tables(df, row_dim, col_dim, factor):
    """Return (slice_dim, {slice_value: formatted DataFrame}). Cell = 'err | iters'."""
    slice_dim = next(d for d in DIMS if d not in (row_dim, col_dim))
    df = df.copy()
    df["cell"] = [_fmt_cell(e, i) for e, i in zip(df["err"], df["iters"])]

    rcol, ccol, scol = COL[row_dim], COL[col_dim], COL[slice_dim]
    tables = {}
    for sv in _dim_order(slice_dim, set(df[scol])):
        sub = df[df[scol] == sv]
        piv = sub.pivot_table(index=rcol, columns=ccol, values="cell", aggfunc="first")
        piv = piv.reindex(index=_dim_order(row_dim, set(piv.index)),
                          columns=_dim_order(col_dim, set(piv.columns)))
        piv = piv.fillna("")
        piv.index.name = DIM_TITLE[row_dim]
        piv.columns.name = DIM_TITLE[col_dim]
        tables[sv] = piv
    return slice_dim, tables


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", choices=DIMS, default="algo", help="dimension for table rows")
    ap.add_argument("--cols", choices=DIMS, default="f", help="dimension for table columns")
    ap.add_argument("--metric", choices=["suboptimality", "train_loss"], default=DEFAULT_METRIC)
    ap.add_argument("--tail-frac", type=float, default=DEFAULT_TAIL_FRAC,
                    help="fraction of the tail used to estimate the asymptotic error")
    ap.add_argument("--factor", type=float, default=DEFAULT_FACTOR,
                    help="reach the plateau to within (factor-1), i.e. cross the band edge "
                         "at factor x L_inf (default 1.05)")
    ap.add_argument("--out", default=TABLE_ROOT, help="output directory for CSVs")
    args = ap.parse_args()
    if args.rows == args.cols:
        ap.error("--rows and --cols must be different dimensions")

    df = build_records(args.metric, args.tail_frac, args.factor)
    if df.empty:
        print(f"No production results found under {PROD_ROOT}/ (did Stage 3 run?)")
        return

    slice_dim, tables = make_tables(df, args.rows, args.cols, args.factor)

    metric_name = "L - L*" if args.metric == "suboptimality" else "train loss"
    print(f"\nCell = (asymptotic error [{metric_name}]) | (#iters to reach {args.factor:g}x it)")
    print(f"asymptotic error = mean of last {args.tail_frac:.0%} of the curve   |   "
          f"rows={DIM_TITLE[args.rows]}, cols={DIM_TITLE[args.cols]}, "
          f"sliced by {DIM_TITLE[slice_dim]}\n")

    os.makedirs(args.out, exist_ok=True)
    # Numeric long form (for downstream use), plus one formatted CSV + Markdown per slice.
    df.to_csv(os.path.join(args.out, f"summary_long_{args.metric}.csv"), index=False)

    md_lines = [
        f"# MNIST / ALIE production tables ({metric_name})",
        "",
        f"Cell = (asymptotic error [{metric_name}]) | (#iters to reach {args.factor:g}x it) — "
        f"asymptotic error = mean of last {args.tail_frac:.0%} of the curve.",
        "",
        f"Rows = {DIM_TITLE[args.rows]}, cols = {DIM_TITLE[args.cols]}, "
        f"sliced by {DIM_TITLE[slice_dim]}.",
        "",
    ]
    for sv, piv in tables.items():
        print(f"=== {DIM_TITLE[slice_dim]} = {sv} ===")
        print(piv.to_string(), "\n")
        base = f"table_{args.rows}_x_{args.cols}__{slice_dim}_{sv}_{args.metric}"
        piv.to_csv(os.path.join(args.out, base + ".csv"))
        piv.to_markdown(os.path.join(args.out, base + ".md"))
        md_lines += [f"## {DIM_TITLE[slice_dim]} = {sv}", "", piv.to_markdown(), ""]

    # Single combined Markdown file with every slice.
    combined = os.path.join(args.out, f"tables_{args.rows}_x_{args.cols}_{args.metric}.md")
    with open(combined, "w") as fh:
        fh.write("\n".join(md_lines))

    print(f"Saved long-form + {len(tables)} per-slice CSV/Markdown table(s) "
          f"and {os.path.basename(combined)} under {args.out}/")


if __name__ == "__main__":
    main()
