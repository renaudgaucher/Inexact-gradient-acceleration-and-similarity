"""Controlled quadratic (linear-regression) testbed for the inexact-gradient methods.

This module builds a synthetic distributed least-squares problem in which every constant that the
paper's theory depends on is set *exactly* by construction:

  * mu  (strong convexity of the global honest loss L_H),
  * L   (smoothness of L_H),  kappa = L / mu,
  * Delta = || nabla^2 L_H - nabla^2 L_1 ||_op   (Hessian similarity of the proxy L_hat = L_1),
  * (G, B)-heterogeneity, with  B^2 = 2 Delta^2 / mu^2  and  G^2 = (2/|H|) sum_i ||grad L_i(x*)||^2.

Each honest client i in H holds m = d synthetic samples (X_i, y_i) whose empirical least-squares
loss, as computed by the framework (``MSELoss`` mean + the L2 term of ``Client._backward_pass``), is

    L_i(x) = (1/m) || X_i x - y_i ||^2 + (lambda/2) ||x||^2 ,   nabla^2 L_i = (2/m) X_i^T X_i + lambda I .

Construction (see ``build_controlled_quadratic``):
  H_i = H0 + E_i + lambda I, with H0 diagonal SPD (spectrum sets kappa), E_i symmetric zero-mean
  across clients with max operator norm rho (sets Delta), and the global optimum pinned at x* = 0
  with per-client gradients g_i = grad L_i(0) drawn zero-mean and scaled to hit a target G.

Because the problem is quadratic the suboptimality is exact:  L_H(x) - L_H(x*) = 1/2 x^T Abar x,
with Abar = mean_i H_i the global Hessian and x* = 0.

The same builder is called by ``byzfl.benchmark.train`` (to generate the per-client data and the
global Hessian used for the exact-suboptimality metric) and by the experiment scripts (to compute
the theory-prescribed step-sizes), so there is a single deterministic source of truth keyed on
``seed``.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------------------------
#  Problem construction
# ---------------------------------------------------------------------------------------------
def _spd_sqrt(M):
    """Symmetric PSD square root of a symmetric PSD matrix M (numpy, float64)."""
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.maximum(w, 0.0))) @ V.T


def build_controlled_quadratic(spec, n_honest, seed=0):
    """Build the per-client synthetic data and the exact problem constants.

    Parameters
    ----------
    spec : dict
        Problem knobs:
          - ``d``      (int)   : dimension / number of samples per client (m = d).
          - ``mu``     (float) : target strong convexity of L_H.
          - ``kappa``  (float) : target condition number (=> L = kappa * mu).
          - ``rho``    (float) : max operator norm of the per-client Hessian perturbation
                                 (=> Delta = rho for the proxy client, index 0). Needs rho < mu - lam.
          - ``G``      (float) : target sqrt((2/|H|) sum_i ||grad L_i(0)||^2).
          - ``lam``    (float) : L2 regularization (weight decay); strong-convexity floor (mu >= lam).
    n_honest : int
        Number of honest clients |H|.
    seed : int
        Deterministic seed for the construction.

    Returns
    -------
    Xs, ys : list[np.ndarray]
        Per-client design matrices (d, d) and targets (d,), float64.
    info : dict
        ``Abar`` (np.ndarray global Hessian), ``lam``, ``d``, and the *realized* constants
        ``mu``, ``L``, ``kappa``, ``Delta``, ``Delta_pairwise``, ``G``, ``B``.
    """
    d = int(spec["d"])
    mu = float(spec["mu"])
    kappa = float(spec["kappa"])
    rho = float(spec["rho"])
    G = float(spec["G"])
    lam = float(spec["lam"])

    L = kappa * mu
    if mu < lam - 1e-12:
        raise ValueError(f"need mu >= lam (got mu={mu}, lam={lam})")
    mu0, L0 = mu - lam, L - lam                                    # H0 spectrum (data part of Abar)
    if rho >= mu0 + 1e-12:
        raise ValueError(f"need rho < mu - lam = {mu0} to keep data Hessians PSD (got rho={rho})")

    rng = np.random.default_rng(seed)
    h0 = np.full(d, mu0) if d == 1 else np.linspace(mu0, L0, d)
    H0 = np.diag(h0)

    # E_i: symmetric, zero-mean across clients, max_i ||E_i||_op == rho.
    Es = [(A + A.T) / 2 for A in (rng.standard_normal((d, d)) for _ in range(n_honest))]
    Es = [E - np.mean(Es, axis=0) for E in Es]
    max_op = max(np.linalg.norm(E, 2) for E in Es) or 1.0
    Es = [rho / max_op * E for E in Es]
    # Make the proxy client (index 0) the max-perturbation one, so Delta == rho exactly.
    # A relabeling (swap) preserves the zero-mean-across-clients property.
    jmax = int(np.argmax([np.linalg.norm(E, 2) for E in Es]))
    Es[0], Es[jmax] = Es[jmax], Es[0]

    Abar = H0 + lam * np.eye(d)
    Hs = [H0 + E + lam * np.eye(d) for E in Es]
    Ds = [H0 + E for E in Es]                                       # data Hessian (2/m) X^T X

    # gradients at x*=0: zero-mean across clients, with (2/|H|) sum ||g_i||^2 = G^2.
    g = rng.standard_normal((n_honest, d))
    g = g - g.mean(axis=0, keepdims=True)
    S = 2.0 / n_honest * float(np.sum(g ** 2))
    if S > 0:
        g = g * np.sqrt(G ** 2 / S)

    m = d
    Xs, ys = [], []
    for i in range(n_honest):
        Xi = np.sqrt(d / 2.0) * _spd_sqrt(Ds[i])                    # (2/m) X_i^T X_i == D_i
        ci = -g[i]                                                  # grad L_i(0) = -c_i = g_i
        yi = (m / 2.0) * np.linalg.solve(Xi.T, ci)                 # (2/m) X_i^T y_i == c_i
        Xs.append(Xi)
        ys.append(yi)

    eig = np.linalg.eigvalsh(Abar)
    info = {
        "Abar": Abar,
        "lam": lam,
        "d": d,
        "mu": float(eig.min()),
        "L": float(eig.max()),
        "kappa": float(eig.max() / eig.min()),
        "Delta": float(np.linalg.norm(Abar - Hs[0], 2)),           # ||Abar - H_1|| = ||E_1|| = rho
        "Delta_pairwise": float(max(np.linalg.norm(Hs[i] - Hs[j], 2)
                                    for i in range(n_honest) for j in range(n_honest))),
        "G": float(np.sqrt(2.0 / n_honest * sum(np.linalg.norm(gi) ** 2 for gi in g))),
    }
    info["B"] = float(np.sqrt(2.0) * info["Delta"] / info["mu"])   # B^2 = 2 Delta^2 / mu^2
    return Xs, ys, info


def make_client_tensors(Xs, ys, device="cpu", dtype=torch.float64):
    """Convert the numpy (X_i, y_i) into per-client (inputs, targets) tensors for MSELoss.

    Targets are shaped (m, 1) to match ``nn.Linear(d, 1)`` outputs.
    """
    tensors = []
    for Xi, yi in zip(Xs, ys):
        x = torch.as_tensor(Xi, dtype=dtype, device=device)
        y = torch.as_tensor(yi, dtype=dtype, device=device).reshape(-1, 1)
        tensors.append((x, y))
    return tensors


# ---------------------------------------------------------------------------------------------
#  Robustness coefficient and theory-prescribed step-sizes
# ---------------------------------------------------------------------------------------------
def huber_nu(f, n_total):
    """(f, nu)-robustness coefficient of the Huber aggregator (Table 1): nu = 2f / (n - 3f).

    n_total is the total number of clients (honest + Byzantine). Returns 0 when f == 0.
    """
    if f <= 0:
        return 0.0
    denom = n_total - 3 * f
    if denom <= 0:
        return float("inf")
    return 2.0 * f / denom


def theory_step_sizes(info, f, n_total, nu=None):
    """Theory-prescribed step-sizes for the four methods, from the known constants.

    Uses mu, L, Delta from ``info`` and the relative-error level alpha = nu * B^2 (nu from the
    Huber coefficient unless overridden). Returns a dict with one sub-dict per method holding the
    scalars each method's config needs.

    Method parametrizations (paper):
      * D-GD (Cor 2.7):                eta = 1 / L.
      * PIGS (Cor 3.8):                eta = 1 / (2 Delta)   (prox step size).
      * Acc. ExtraGrad (Cor 4.2, Delta=L): gamma=mu, theta=1/(2 sqrt2 L),
                                       tau=min(theta*mu/(16 alpha), sqrt(theta*mu/48)),
                                       eta=min(1/(2 mu), theta/(24 tau)),  momentum = eta*gamma.
      * Acc. ExtraGrad Prox (Thm 4.1): same with theta = 1/(2 sqrt2 Delta).
    """
    mu, L, Delta, B = info["mu"], info["L"], info["Delta"], info["B"]
    if nu is None:
        nu = huber_nu(f, n_total)
    alpha = nu * B ** 2

    def accel(theta):
        tau_stab = np.sqrt(theta * mu / 48.0)
        tau = tau_stab if alpha <= 0 else min(theta * mu / (16.0 * alpha), tau_stab)
        eta = min(1.0 / (2.0 * mu), theta / (24.0 * tau))
        return {"theta": float(theta), "tau": float(tau),
                "fast_step_size": float(eta), "gamma": float(mu),
                "momentum": float(eta * mu)}

    return {
        "nu": float(nu),
        "alpha": float(alpha),
        "gd":   {"eta": 1.0 / L},
        "pigs": {"eta": 1.0 / (2.0 * Delta)},
        "accextragrad":     accel(1.0 / (2.0 * np.sqrt(2.0) * L)),      # L_hat = 0  => Delta = L
        "accextragradprox": accel(1.0 / (2.0 * np.sqrt(2.0) * Delta)),
    }
