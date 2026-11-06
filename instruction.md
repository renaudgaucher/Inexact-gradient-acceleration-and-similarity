All three orientations, both metrics, and CSV output work. Created analysis_mnist_alie_table.py.

# What it does
Each cell is asymptotic error | #iterations:

- asymptotic error = the plateau, estimated as the mean over the last --tail-frac (default 10%) of the seed-averaged curve. Metric is L - L* by default (--metric train_loss for raw loss), reusing reference_optimum / prod_run_folder / stacked_train_loss from the common module so it stays consistent with the plots.
- #iterations = first iteration that reaches the ±(factor−1) band around the plateau (|value − err| ≤ |factor−1|·|err|, default factor 1.01). Since train loss is logged per step, the array index is the iteration number.


# Choosing rows/columns
You pick any two of the three varying parameters for --rows / --cols; the third is auto-detected and sliced (one table per value):


> python analysis_mnist_alie_table.py --rows algo --cols f      # one table per heterogeneity
> python analysis_mnist_alie_table.py --rows het  --cols algo   # one table per f
> python analysis_mnist_alie_table.py --rows f    --cols het    # one table per algorithm

Extra flags: --metric {suboptimality,train_loss}, --tail-frac, --factor, --out. It prints the tables and writes a numeric long-form CSV plus one formatted per-slice CSV under ./tables/mnist_alie_prod/.