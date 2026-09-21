#!/usr/bin/env python
"""Superconductor: CRC, AA-CRC (authors' objective; linear and RF-leaf feature maps) and ReCIRC.

AA-CRC arms (--aacrc-basis {linear, rf, both}; default both)
------------------------------------------------------------
Both arms call J / J_prime from the pinned authors' module (SHA-256 checked) and
use the exact asymmetric-loss encoding with weights (0.2, 0.8): five auxiliary
positive labels (one lower-tail, four upper-tail). The learned threshold is
u(x) = Phi(x) @ theta and the interval multiplier is max(0, -log u) for u > 0.
theta is always fitted on the independent calibration split C.

* AA-CRC (RF leaves): the tabular procedure of Blot et al. (AISTATS 2025,
  Sec. 3, Algorithm 1). A random forest is trained on D to predict the minimal
  covering multiplier of each point (log1p scale), computed from OUT-OF-BAG QRF
  quantiles so that the D residuals are not shrunk by in-sample fitting.
  Phi(x) is the one-hot vector of the leaves reached by x. No ridge (as in the
  authors' notebook); box constraints keep u in [exp(-LAM_MAX), 1], so the
  multiplier never exceeds LAM_MAX and no deployment truncation is needed.
* AA-CRC (linear): Phi(x) = [1, standardized covariates]. The ridge penalty
  excludes the intercept by default, so by Theorem 1 of Blot et al. the
  marginal level is not shifted by -2*rho*theta_0 (--aacrc-ridge-intercept
  restores the authors' full ridge). Deployment multipliers are truncated at
  LAM_MAX (--no-aacrc-truncate disables it).

--aacrc-features {x, feats} chooses the covariates given to both arms: the raw
physical covariates (authors' choice) or the same feature map used by ReCIRC.

This is a task adaptation of the authors' objective, not an unchanged
reproduction of their regression experiment; no additional validity claim.

CRC and ReCIRC keep the finite lambda grid; ReCIRC keeps the fixed 201-point
budget grid on [0, 1]. All CRC-type bounds now use B = max(W_NEG, W_POS).
--recirc-oob-context builds ReCIRC's context losses from OOB QRF quantiles.

Evaluation retains the original 4x4 feature-bin slices (target-selected,
exploratory; never used for fitting). For the RF arm, the risk inside its own
leaves on the test split is also reported (own_group_worst).

--self-check needs only NumPy/SciPy (plus scikit-learn for the RF check);
--no-recirc omits ReCIRC. Packages are never installed automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

AACRC_COMMIT = "64504c011ac2db910e258037e48170a63381b5e6"
AACRC_SOURCE_SHA256 = "9b3cffcb45f2e9a74f467ec94a00be7fcf2a24ec85f2dfe56f01d7fbd4a51315"
AACRC_REPO = Path(__file__).resolve().parents[1] / "AA-CRC"
AACRC_INTEGRATION = "serial"
AACRC_RIDGE = 0.01
AACRC_MAXITER = 200
AACRC_OUTPUT_DIR = None
AACRC_MODULE = None

# Loss and lambda-grid constants are defined here (before --self-check runs).
W_NEG, W_POS = 0.2, 0.8
N_LAM = 80
LAM_MAX = 4.0

# RF leaf feature map (Blot et al., Algorithm 1).
RF_TREES = 5
RF_DEPTH = 5
RF_MIN_LEAF = 200
RF_LEAF_NMIN_DIAG = 50


def load_original_aacrc(repo, integration="serial"):
    """Load the pinned authors' module; optionally select its own serial helper."""
    if integration not in {"serial", "parallel"}:
        raise ValueError("integration must be 'serial' or 'parallel'.")
    repo = Path(repo).resolve()
    path = repo / "multiaccurate_cp" / "utils" / "multiaccurate.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"AA-CRC source missing: {path}. Clone vincentblot28/AA-CRC, "
            f"checkout {AACRC_COMMIT}, and pass --aacrc-repo PATH."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != AACRC_SOURCE_SHA256:
        raise RuntimeError(f"Unexpected AA-CRC source SHA-256: {digest}; expected {AACRC_SOURCE_SHA256}.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    module = importlib.import_module("multiaccurate_cp.utils.multiaccurate")
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("Another multiaccurate_cp module is already loaded; use a fresh Python process.")
    if not hasattr(module, "_original_parallel_integrals"):
        module._original_parallel_integrals = module._I_vec_multi_proc
    module._I_vec_multi_proc = (
        module._I_vec_multi_proc2 if integration == "serial" else module._original_parallel_integrals
    )
    return module


def auxiliary_tail_labels(y, median, s_neg, s_pos):
    """Exact 1:4 binary-label reduction of the (0.2, 0.8) asymmetric loss."""
    y, median, s_neg, s_pos = [np.asarray(v, dtype=float) for v in (y, median, s_neg, s_pos)]
    if not (y.ndim == 1 and y.shape == median.shape == s_neg.shape == s_pos.shape):
        raise ValueError("Expected equally sized one-dimensional response and QRF arrays.")
    if not all(np.isfinite(v).all() for v in (y, median, s_neg, s_pos)):
        raise ValueError("Responses and QRF arrays must be finite.")
    if np.any(s_neg <= 0) or np.any(s_pos <= 0):
        raise ValueError("QRF scales must be strictly positive.")
    residuals = np.column_stack([(median - y) / s_neg, (y - median) / s_pos])
    tail_scores = np.full(residuals.shape, np.inf)
    positive = residuals > 0
    tail_scores[positive] = np.exp(-residuals[positive])
    if np.any(tail_scores[positive] == 0):
        raise FloatingPointError("Exponential score underflow; rescale/review the QRF scaffold before fitting.")
    scores = tail_scores[:, [0, 1, 1, 1, 1]]
    labels = [np.ones((5, 1), dtype=float) for _ in y]
    return labels, [row[:, None].copy() for row in scores]


def multiplier_from_aacrc_threshold(u, lam_grid=None, truncate=False, snap_to_grid=False):
    """Map the AA-CRC score threshold u to the interval multiplier.

    With truncate=False this is the raw mapping the fitted objective implies,
    and u <= 0 yields an infinite multiplier. That branch is the authors'
    fully protective endpoint, finite in a bounded universe but unbounded for
    real-valued responses.

    With truncate=True the multiplier is clipped to the CRC/ReCIRC grid range,
    so u <= 0 deploys the most protective rule that the shared family actually
    contains. snap_to_grid additionally rounds to the nearest grid point, which
    also charges AA-CRC the discretization the other two arms pay.

    Returns (lam, diagnostics).
    """
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all():
        raise ValueError("AA-CRC score thresholds must be finite.")
    if lam_grid is None:
        lam_grid = np.linspace(0.0, LAM_MAX, N_LAM)
    lam_grid = np.asarray(lam_grid, dtype=float)
    lam_lo, lam_hi = float(lam_grid[0]), float(lam_grid[-1])

    lam_raw = np.full(u.shape, np.inf)
    positive = u > 0
    lam_raw[positive] = np.maximum(0.0, -np.log(u[positive]))

    lam = lam_raw
    if truncate:
        lam = np.clip(lam_raw, lam_lo, lam_hi)
        if snap_to_grid:
            lam = lam_grid[np.abs(lam_grid[None, :] - lam[:, None]).argmin(axis=1)]

    finite_raw = lam_raw[np.isfinite(lam_raw)]
    diagnostics = {
        "truncate": bool(truncate),
        "snap_to_grid": bool(snap_to_grid),
        "lam_max": lam_hi,
        "rate_u_nonpositive": float((u <= 0).mean()),
        "rate_would_exceed_lam_max": float((lam_raw > lam_hi).mean()),
        "rate_at_lam_max": float((lam >= lam_hi - 1e-12).mean()),
        "lam_raw_max_finite": float(finite_raw.max()) if finite_raw.size else float("nan"),
        "lam_raw_median_finite": float(np.median(finite_raw)) if finite_raw.size else float("nan"),
    }
    return lam, diagnostics


def evaluate_aacrc_intervals(y, median, s_neg, s_pos, u, lam_grid=None,
                             truncate=False, snap_to_grid=False):
    """Interval loss and width implied by the AA-CRC thresholds.

    Defaults reproduce the untruncated mapping, which is what the fit-time
    encoding-equivalence check must verify. Deployment passes truncate=True.
    """
    lam, diagnostics = multiplier_from_aacrc_threshold(
        u, lam_grid=lam_grid, truncate=truncate, snap_to_grid=snap_to_grid
    )
    lo = np.asarray(median, float) - lam * np.asarray(s_neg, float)
    hi = np.asarray(median, float) + lam * np.asarray(s_pos, float)
    loss = W_NEG * (np.asarray(y) < lo) + W_POS * (np.asarray(y) > hi)
    return loss, hi - lo, lam, diagnostics


def _aacrc_objective(module, labels, scores, phi, alpha, n, ridge, ridge_mask):
    """Authors' J / J_prime (no internal regularization) plus an external ridge.

    The penalty acts on the coordinates selected by ``ridge_mask``. With a mask of
    ones this equals the authors' ridge; with ridge_mask[0] = 0 the intercept is
    free and, by Theorem 1 of Blot et al., the marginal level is not shifted.
    """
    mask = np.asarray(ridge_mask, dtype=float)

    def objective(theta):
        value = module.J(theta, labels, scores, phi, alpha, n, None, None)
        return float(value + ridge * np.sum(mask * theta ** 2))

    def gradient(theta):
        grad = np.asarray(module.J_prime(theta, labels, scores, phi, alpha, n, None, None), dtype=float)
        return grad + 2.0 * ridge * mask * theta

    return objective, gradient


def fit_aacrc_core(phi, y_C, median_C, s_neg_C, s_pos_C, *, alpha, ridge, ridge_mask,
                   theta0, bounds=None, maxiter=200, module=None, diagnostic_path=None,
                   extra_diag=None):
    """Optimize the authors' AA-CRC objective for a given feature matrix on C."""
    if module is None:
        module = load_original_aacrc(AACRC_REPO, AACRC_INTEGRATION)
    phi = np.asarray(phi, dtype=float)
    n = len(y_C)
    if not 0 < alpha < 1 or n <= 1.0 / alpha:
        raise ValueError("Need 0 < alpha < 1 and len(C) > 1/alpha for this AA-CRC fit.")
    if ridge < 0 or maxiter < 1:
        raise ValueError("ridge must be nonnegative and maxiter positive.")
    if phi.shape[0] != n or not np.isfinite(phi).all():
        raise ValueError("Feature matrix must be finite with one row per calibration point.")
    labels, scores = auxiliary_tail_labels(y_C, median_C, s_neg_C, s_pos_C)
    objective, gradient = _aacrc_objective(module, labels, scores, phi, alpha, n, ridge, ridge_mask)
    theta = np.asarray(theta0, dtype=float).copy()
    attempts = []
    start = time.time()
    success = False
    result = None
    for iterations in (maxiter, 3 * maxiter):
        result = minimize(objective, theta, method="SLSQP", jac=gradient, bounds=bounds,
                          tol=1e-8, options={"maxiter": iterations, "disp": False})
        success = bool(result.success and np.isfinite(result.x).all() and np.isfinite(result.fun))
        attempts.append({"success": success, "status": int(result.status),
                         "message": str(result.message), "iterations": int(result.nit),
                         "objective": float(result.fun) if np.isfinite(result.fun) else None})
        if success:
            break
        if np.isfinite(result.x).all():
            theta = result.x.copy()
    diag = {"source_commit": AACRC_COMMIT, "source_sha256": AACRC_SOURCE_SHA256,
            "fit_split": "C", "n_fit": n, "n_params": int(phi.shape[1]),
            "n_auxiliary_labels_per_observation": 5, "ridge": float(ridge),
            "ridge_on_intercept": bool(np.asarray(ridge_mask)[0] > 0) if ridge > 0 else False,
            "bounds": None if bounds is None else [list(bounds[0])],
            "success": success, "attempts": attempts, "elapsed_seconds": time.time() - start}
    if extra_diag:
        diag.update(extra_diag)
    if success:
        theta = result.x.copy()
        u = phi @ theta
        if np.any(u < -module.INF_BORN_INT):
            success = False
            diag["success"] = False
            diag["error"] = "Fitted thresholds reached the original objective's lower truncation."
        loss, _, _, _ = evaluate_aacrc_intervals(
            y_C, median_C, s_neg_C, s_pos_C, u, truncate=False
        )
        encoded_loss = module._I_prime_list(labels, scores, np.maximum(u, 0.0), alpha, n) + alpha - 1.0 / n
        diag.update({"theta": theta.tolist(),
                     "fit_risk": float(loss.mean()),
                     "stationarity_target": float(alpha - 1.0 / n),
                     "encoding_max_abs_error": float(np.max(np.abs(loss - encoded_loss))),
                     "threshold_min": float(u.min()), "threshold_max": float(u.max())})
        if bounds is not None:
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            at_bound = np.isclose(theta, lo, rtol=0, atol=1e-9) | np.isclose(theta, hi, rtol=0, atol=1e-9)
            diag["rate_theta_at_bound"] = float(at_bound.mean())
        if diag["encoding_max_abs_error"] > 1e-10:
            success = False
            diag["success"] = False
            diag["error"] = "Interval/auxiliary-label losses disagree (check numerical boundary ties)."
    if diagnostic_path is not None:
        Path(diagnostic_path).write_text(json.dumps(diag, indent=2, allow_nan=False))
    if not success:
        raise RuntimeError("Original AA-CRC optimization failed: " + json.dumps(diag))
    return {"theta": theta, "diagnostics": diag, "phi_fit": phi}


def fit_original_aacrc(X_D, X_C, y_C, median_C, s_neg_C, s_pos_C,
                       alpha=0.1, ridge=0.01, maxiter=200, module=None,
                       diagnostic_path=None, ridge_intercept=False):
    """Linear AA-CRC: Phi = [1, covariates standardized with D statistics], theta on C."""
    center = np.asarray(X_D, float).mean(axis=0)
    scale = np.asarray(X_D, float).std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    phi = np.column_stack([np.ones(len(y_C)), (np.asarray(X_C, float) - center) / scale])
    theta0 = np.zeros(phi.shape[1])
    theta0[0] = 0.5
    mask = np.ones(phi.shape[1])
    if not ridge_intercept:
        mask[0] = 0.0
    fit = fit_aacrc_core(
        phi, y_C, median_C, s_neg_C, s_pos_C, alpha=alpha, ridge=ridge, ridge_mask=mask,
        theta0=theta0, maxiter=maxiter, module=module, diagnostic_path=diagnostic_path,
        extra_diag={"feature_map": "linear", "feature_standardization_split": "D"},
    )
    fit.update({"center": center, "scale": scale, "kind": "linear"})
    return fit


def predict_original_aacrc(fit, X):
    phi = np.column_stack([np.ones(len(X)), (np.asarray(X, float) - fit["center"]) / fit["scale"]])
    return phi @ fit["theta"]


# -----------------------------------------------------------------------------
# AA-CRC with random-forest leaf indicators (Blot et al., Algorithm 1)
# -----------------------------------------------------------------------------


def minimal_cover_multiplier(y, median, s_neg, s_pos):
    """Smallest interval multiplier that covers y (zero loss); RF target on D."""
    y, median, s_neg, s_pos = [np.asarray(v, dtype=float) for v in (y, median, s_neg, s_pos)]
    return np.maximum.reduce([(median - y) / s_neg, (y - median) / s_pos, np.zeros_like(y)])


class LeafFeatureMap:
    """Phi(x) = one-hot indicators of the leaves reached by x in each tree."""

    def __init__(self, Z_res, target, n_trees=RF_TREES, max_depth=RF_DEPTH,
                 min_leaf=RF_MIN_LEAF, seed=0):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder

        Z_res = np.asarray(Z_res, dtype=float)
        self.rf = RandomForestRegressor(
            n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_leaf,
            n_jobs=-1, random_state=seed,
        ).fit(Z_res, np.asarray(target, dtype=float))
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.encoder = encoder.fit(self.rf.apply(Z_res))
        self.n_trees = int(n_trees)

    def __call__(self, Z):
        return self.encoder.transform(self.rf.apply(np.asarray(Z, dtype=float))).astype(float)

    @property
    def n_leaves(self):
        return int(sum(len(c) for c in self.encoder.categories_))


def fit_aacrc_rf(leaf_map, Z_C, y_C, median_C, s_neg_C, s_pos_C, *, alpha, lam_init,
                 lam_max=LAM_MAX, maxiter=200, module=None, diagnostic_path=None):
    """RF-leaf AA-CRC: theta on C, no ridge, box constraints.

    Each tree contributes exactly one active indicator, so u(x) is the sum of one
    coefficient per tree. Bounds theta_j in [exp(-lam_max)/T, 1/T] keep u in
    [exp(-lam_max), 1], i.e. the multiplier -log(u) in [0, lam_max]: the same
    family as CRC/ReCIRC, with no deployment truncation.
    """
    phi = leaf_map(Z_C)
    T = leaf_map.n_trees
    lower, upper = float(np.exp(-lam_max)) / T, 1.0 / T
    theta0 = np.full(phi.shape[1], np.clip(np.exp(-lam_init) / T, lower, upper))
    fit = fit_aacrc_core(
        phi, y_C, median_C, s_neg_C, s_pos_C, alpha=alpha, ridge=0.0,
        ridge_mask=np.zeros(phi.shape[1]), theta0=theta0,
        bounds=[(lower, upper)] * phi.shape[1], maxiter=maxiter, module=module,
        diagnostic_path=None,
        extra_diag={"feature_map": "rf_leaves", "n_trees": T, "n_leaves": leaf_map.n_leaves,
                    "rf_max_depth": leaf_map.rf.max_depth,
                    "rf_min_samples_leaf": leaf_map.rf.min_samples_leaf},
    )
    u = phi @ fit["theta"]
    loss, _, _, _ = evaluate_aacrc_intervals(y_C, median_C, s_neg_C, s_pos_C, u, truncate=False)
    counts = phi.sum(axis=0)
    leaf_risk = (phi * loss[:, None]).sum(axis=0) / np.maximum(counts, 1.0)
    populated = counts >= RF_LEAF_NMIN_DIAG
    fit["diagnostics"].update({
        "fit_worst_leaf_risk": float(leaf_risk[populated].max()) if populated.any() else None,
        "fit_n_populated_leaves": int(populated.sum()),
    })
    if diagnostic_path is not None:
        Path(diagnostic_path).write_text(json.dumps(fit["diagnostics"], indent=2, allow_nan=False))
    fit.update({"leaf_map": leaf_map, "kind": "rf"})
    return fit


def predict_aacrc_rf(fit, Z):
    return fit["leaf_map"](Z) @ fit["theta"]


def own_group_worst(loss, phi, nmin=RF_LEAF_NMIN_DIAG):
    """Worst test risk over the fitted groups (leaves) with at least nmin points."""
    counts = phi.sum(axis=0)
    risk = (phi * np.asarray(loss, float)[:, None]).sum(axis=0) / np.maximum(counts, 1.0)
    ok = counts >= nmin
    return float(risk[ok].max()) if ok.any() else float("nan")


def check_original_aacrc(repo):
    """Offline loss-identity, quadrature, gradient and optimizer checks."""
    module = load_original_aacrc(repo, "serial")
    rng = np.random.default_rng(819)
    n = 80
    X_D, X_C = rng.normal(size=(120, 3)), rng.normal(size=(n, 3))
    median = rng.normal(size=n)
    s_neg, s_pos = rng.uniform(0.5, 2, size=(2, n))
    y = median + rng.normal(size=n) * 1.5
    y[0] = median[0]
    labels, scores = auxiliary_tail_labels(y, median, s_neg, s_pos)
    alpha = 0.2
    max_loss_error = 0.0
    for threshold in (-0.2, 0, 0.01, 0.2, 0.8, 1, 1.5, 3):
        u = np.full(n, threshold)
        direct, widths, _, _ = evaluate_aacrc_intervals(
            y, median, s_neg, s_pos, u, truncate=False
        )
        encoded = module._I_prime_list(labels, scores, np.maximum(0, u), alpha, n) + alpha - 1 / n
        err = float(np.max(np.abs(direct - encoded)))
        max_loss_error = max(max_loss_error, err)
        assert err < 1e-12, (threshold, err)
        assert np.all(widths >= 0)
    phi = np.column_stack([np.ones(n), X_C])
    theta = np.array([0.4, 0.02, -0.01, 0.03])
    u = phi @ theta
    ridge = 0.01
    score_table = np.stack([row.ravel() for row in scores])
    target = alpha - 1 / n
    exact = np.mean(np.maximum(u[:, None] - score_table, 0).mean(axis=1) - u * target)
    exact -= module.INF_BORN_INT * target
    exact += ridge * np.sum(theta ** 2)
    numeric = module.J(theta, labels, scores, phi, alpha, n, "ridge", ridge)
    quadrature_error = abs(float(numeric - exact))
    assert quadrature_error < 0.005, quadrature_error
    direct, _, _, _ = evaluate_aacrc_intervals(y, median, s_neg, s_pos, u, truncate=False)
    expected_gradient = np.mean(phi * (direct - target)[:, None], axis=0) + 2 * ridge * theta
    gradient = module.J_prime(theta, labels, scores, phi, alpha, n, "ridge", ridge)
    assert np.allclose(gradient, expected_gradient, atol=1e-12)
    fit = fit_original_aacrc(X_D, X_C, y, median, s_neg, s_pos, alpha=alpha, module=module)
    test_u = predict_original_aacrc(fit, X_C[:8])
    assert test_u.shape == (8,) and np.isfinite(test_u).all()
    # Intercept-free ridge: external penalty must match the authors' ridge off the intercept.
    objective, grad_fn = _aacrc_objective(module, labels, scores, phi, alpha, n, ridge,
                                          np.r_[0.0, np.ones(phi.shape[1] - 1)])
    manual = module.J(theta, labels, scores, phi, alpha, n, None, None) + ridge * np.sum(theta[1:] ** 2)
    assert abs(objective(theta) - manual) < 1e-12
    assert np.allclose(grad_fn(theta)[0], expected_gradient[0] - 2 * ridge * theta[0], atol=1e-12)
    rf_report = "skipped (scikit-learn unavailable)"
    try:
        leaf_map = LeafFeatureMap(X_D, rng.gamma(2.0, size=len(X_D)), n_trees=3, max_depth=2,
                                  min_leaf=10, seed=0)
        rf_fit = fit_aacrc_rf(leaf_map, X_C, y, median, s_neg, s_pos, alpha=alpha,
                              lam_init=1.0, lam_max=4.0, module=module)
        u_rf = predict_aacrc_rf(rf_fit, X_C)
        assert np.all(u_rf >= np.exp(-4.0) - 1e-9) and np.all(u_rf <= 1.0 + 1e-9)
        rf_report = {"n_leaves": leaf_map.n_leaves,
                     "fit_risk": rf_fit["diagnostics"]["fit_risk"],
                     "rate_theta_at_bound": rf_fit["diagnostics"]["rate_theta_at_bound"]}
    except ImportError:
        pass
    print(json.dumps({"loss_identity_max_error": max_loss_error,
                      "official_quadrature_abs_error": quadrature_error,
                      "gradient_check": "passed", "optimizer": fit["diagnostics"],
                      "rf_leaf_check": rf_report}, indent=2, default=str))


# This check deliberately runs before importing QRF, Torch or TabICL.
if __name__ == "__main__" and "--self-check" in sys.argv:
    check_parser = argparse.ArgumentParser(description="Offline validation of the original AA-CRC adaptation.")
    check_parser.add_argument("--self-check", action="store_true")
    check_parser.add_argument("--aacrc-repo", type=Path, default=AACRC_REPO)
    check_args = check_parser.parse_args()
    check_original_aacrc(check_args.aacrc_repo)
    raise SystemExit(0)


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from tabicl import TabICLRegressor

    HAS_TABICL = True
except Exception as _tabicl_import_error:  # pragma: no cover - environment dependent
    HAS_TABICL = False
    print("aviso: tabicl indisponível —", _tabicl_import_error)


# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------

ALPHA = 0.10
N_TRIALS = 20
BASE_SEED = 42

# Asymmetric loss weights W_NEG, W_POS are defined at the top of the file.

# True bound of the asymmetric loss: it never exceeds max(W_NEG, W_POS).
# The CRC correction uses B / (n + 1); B = 1.0 is valid but conservative.
B_LOSS = max(W_NEG, W_POS)

# AA-CRC deployment: the authors' objective lives on the score coordinate u.
# u <= 0 means "the most protective rule available" -- finite in their setting
# (all pixels), unbounded here (the whole real line). Truncating the recovered
# multiplier at LAM_MAX puts AA-CRC in the SAME family as CRC and ReCIRC.
# This affects deployment only; the fit-time encoding check stays untruncated.
AACRC_TRUNCATE = True
AACRC_SNAP_TO_GRID = False

# QRF quantile levels used to build the (lower, median, upper) scaffold.
Q_LO, Q_MED, Q_HI = 0.05, 0.50, 0.95

# Lambda grid used by CRC and ReCIRC (multiplier applied to the asymmetric scales).
# N_LAM and LAM_MAX are defined at the top of the file.
EPS_SCALE = 1e-3

# Number of lambda anchors on which ReCIRC actually fits a risk regressor;
# the remaining grid points come from monotone PCHIP interpolation.
N_LAM_TRAIN = 16

# Pre-calibration budget grid on [0, 1], fixed for every split (budgets above
# B_LOSS are never binding).
# Calibration losses select a budget from this grid; they never define it.
A_GRID = np.linspace(0.0, 1.0, 201)
A_GRID.setflags(write=False)

# Split fractions: D = risk/context, C = calibration, T = test (the remainder).
FRAC_D, FRAC_C = 0.40, 0.30

# QRF hyperparameters.
N_TREES_QRF = 200
MAX_DEPTH_QRF = 12
MIN_LEAF_QRF = 40

# Minimum slice size for a slice to enter the slice statistics.
SLICE_NMIN = 50

# Number of quantile bins per slicing feature (slices are the 4x4 interactions).
N_SLICE_BINS = 2

# Risk regressor backbone for R(x, lambda).
RISK_BACKEND = "tabicl"  # "tabicl" (foundation model) or "histgb"
TABICL_ESTIMATORS = 4
N_D_TABICL = 4000  # TabICL context size (attention is quadratic in the context)
BATCH_ROWS_TABICL = 4000

try:
    import torch

    TABICL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    TABICL_DEVICE = "cpu"

# Detect Google Drive mounted (Colab) and use as default
if os.path.isdir("/content/drive"):
    OUT_DIR = (
        "/content/drive/MyDrive/PythonReCIRC/results/experiment_mechanism_superconductor_official_aacrc_rf_leaves"
    )
else:
    OUT_DIR = "mechanism_superconductor_official_aacrc_rf_leaves_results"

# Method labels (kept stable across CSV outputs and figures).
METHOD_CRC = "CRC"
METHOD_AACRC_LIN = "AA-CRC (linear)"
METHOD_AACRC_RF = "AA-CRC (RF leaves)"
METHOD_RECIRC = "ReCIRC"
METHODS_ORDER = [METHOD_CRC, METHOD_AACRC_LIN, METHOD_AACRC_RF, METHOD_RECIRC]  # filtered in main()

PALETTE = {
    METHOD_CRC: "#888888",
    METHOD_AACRC_LIN: "#d62728",
    METHOD_AACRC_RF: "#CC79A7",
    METHOD_RECIRC: "#1f77b4",
}

# Set from the command line in main().
AACRC_BASIS = "both"
AACRC_FEATURES = "x"
AACRC_RIDGE_INTERCEPT = False
RECIRC_OOB_CONTEXT = True


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

SUPER_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
SUPER_FALLBACK = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"


def load_superconductor(cache_file: str = "./data_superconductor/train.csv") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Download (once) and load the UCI Superconductivity dataset.

    The archive is fetched from the primary UCI URL with a legacy fallback, then
    extracted into a temporary directory; only ``train.csv`` is kept and cached
    locally so that repeated runs do not hit the network.
    """
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(cache_file):
        with tempfile.TemporaryDirectory(prefix="super_") as wd:
            zp = os.path.join(wd, "s.zip")
            for url in (SUPER_URL, SUPER_FALLBACK):
                try:
                    print("baixando", url, "...")
                    urllib.request.urlretrieve(url, zp)
                    break
                except Exception as exc:
                    print("  falhou:", str(exc)[:100])
            else:
                raise RuntimeError("Não consegui baixar Superconductor; passe --data-file com um CSV local.")
            with zipfile.ZipFile(zp) as zf:
                members = [name for name in zf.namelist() if Path(name).name == "train.csv"]
                if len(members) != 1:
                    raise RuntimeError("Esperado exatamente um train.csv no arquivo baixado.")
                with zf.open(members[0]) as source:
                    Path(cache_file).write_bytes(source.read())

    df = pd.read_csv(cache_file)
    if "critical_temp" not in df or df.shape[1] < 3 or len(df) == 0:
        raise ValueError("CSV precisa de critical_temp, ao menos duas covariáveis e observações.")
    y = df["critical_temp"].to_numpy(np.float32)
    X = df.drop(columns=["critical_temp"]).to_numpy(np.float32)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("As covariáveis e critical_temp devem ser finitas.")
    names = list(df.drop(columns=["critical_temp"]).columns)
    print(
        f"Superconductor: {X.shape[0]} × {X.shape[1]} | temp_crit med={np.median(y):.1f} máx={y.max():.1f}"
    )
    return X, y, names


def quantile_bins(v: np.ndarray, n_bins: int = N_SLICE_BINS) -> np.ndarray:
    """Assign each value to an equal-mass quantile bin."""
    e = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return np.clip(np.digitize(v, e[1:-1]), 0, n_bins - 1)


def build_slice_function(
    X: np.ndarray, y: np.ndarray, names: Sequence[str], n_bins: int = N_SLICE_BINS
) -> Tuple[Callable[[np.ndarray], Dict[str, np.ndarray]], List[str]]:
    """Build the evaluation slices from the two features most correlated with the target.

    Each of the two features is binned into ``n_bins`` quantile bins; the slices
    are the resulting bin interactions. Feature selection uses full-data labels;
    these exploratory slices are never exposed to any of the methods.
    """
    corr = np.array([abs(np.corrcoef(X[:, k], y)[0, 1]) for k in range(X.shape[1])])
    top2 = np.argsort(corr)[-2:]
    slice_features = [names[k] for k in top2]
    print("features p/ slices:", slice_features)

    b1_all = quantile_bins(X[:, top2[0]], n_bins)
    b2_all = quantile_bins(X[:, top2[1]], n_bins)

    def slice_fn(idx_T: np.ndarray) -> Dict[str, np.ndarray]:
        b1, b2 = b1_all[idx_T], b2_all[idx_T]
        return {f"{a}_{b}": (b1 == a) & (b2 == b) for a in range(n_bins) for b in range(n_bins)}

    return slice_fn, slice_features


# -----------------------------------------------------------------------------
# Base predictor, loss construction and features
# -----------------------------------------------------------------------------


def fit_qrf(
    X_all: np.ndarray, y_all: np.ndarray, idx_D: np.ndarray, seed: int
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit the QRF scaffold on D and predict the (lo, med, hi) quantiles for all rows.

    Returns two triples: the usual predictions for all rows, and a copy in which
    the D rows are replaced by out-of-bag predictions (residuals on D are then
    not shrunk by in-sample fitting). The lower and upper quantiles are clipped
    against the median so that the asymmetric scales are non-negative.
    """
    try:
        from quantile_forest import RandomForestQuantileRegressor
    except ImportError as exc:
        raise RuntimeError("Instale quantile-forest no ambiente para executar o experimento.") from exc

    qrf = RandomForestQuantileRegressor(
        n_estimators=N_TREES_QRF,
        max_depth=MAX_DEPTH_QRF,
        min_samples_leaf=MIN_LEAF_QRF,
        n_jobs=-1,
        random_state=seed,
    ).fit(X_all[idx_D], y_all[idx_D])

    Q = qrf.predict(X_all, quantiles=[Q_LO, Q_MED, Q_HI])
    Q_oob = Q.copy()
    Q_oob[idx_D] = qrf.predict(X_all[idx_D], quantiles=[Q_LO, Q_MED, Q_HI], oob_score=True)

    def unpack(M):
        lo = np.minimum(M[:, 0], M[:, 1]).astype(np.float32)
        hi = np.maximum(M[:, 2], M[:, 1]).astype(np.float32)
        return lo, M[:, 1].astype(np.float32), hi

    return unpack(Q), unpack(Q_oob)


def build_loss(
    y: np.ndarray, q_lo: np.ndarray, q_med: np.ndarray, q_hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the loss and width matrices over the lambda grid.

    Returns:
        LOSS: (n, N_LAM) asymmetric miscoverage loss for each row and lambda.
        WIDTH: (n, N_LAM) interval width in kelvin.
        lam: the lambda grid.
        s_neg, s_pos: the lower/upper asymmetric scales.
    """
    s_neg = np.maximum(q_med - q_lo, EPS_SCALE).astype(np.float32)
    s_pos = np.maximum(q_hi - q_med, EPS_SCALE).astype(np.float32)
    lam = np.linspace(0, LAM_MAX, N_LAM).astype(np.float32)

    lo_b = q_med[:, None] - lam[None, :] * s_neg[:, None]
    hi_b = q_med[:, None] + lam[None, :] * s_pos[:, None]

    LOSS = (W_NEG * (y[:, None] < lo_b) + W_POS * (y[:, None] > hi_b)).astype(np.float32)
    WIDTH = (hi_b - lo_b).astype(np.float32)
    return LOSS, WIDTH, lam, s_neg, s_pos


def feats(X: np.ndarray, q_med: np.ndarray, s_neg: np.ndarray, s_pos: np.ndarray) -> np.ndarray:
    """Feature map used by the ReCIRC risk regressor: raw covariates + scale summaries."""
    w0 = s_neg + s_pos
    return np.column_stack(
        [X, q_med, s_neg, s_pos, w0, s_pos / (s_neg + s_pos), np.log1p(w0)]
    ).astype(np.float32)


# -----------------------------------------------------------------------------
# CRC calibration
# -----------------------------------------------------------------------------


def crc_upper_bound(mean_loss: np.ndarray, n: int, B: float = B_LOSS) -> np.ndarray:
    """Finite-sample CRC upper bound (n / (n + 1)) * Rhat + B / (n + 1)."""
    return (n / (n + 1.0)) * mean_loss + B / (n + 1.0)


def crc_global(loss_cal: np.ndarray, alpha: float = ALPHA, B: float = B_LOSS) -> int:
    """Marginal CRC: smallest lambda index whose CRC bound is below alpha."""
    n = loss_cal.shape[0]
    rhat = loss_cal.mean(0)
    bound = crc_upper_bound(rhat, n=n, B=B)
    v = np.where(bound <= alpha)[0]
    return int(v[0]) if len(v) else loss_cal.shape[1] - 1


# -----------------------------------------------------------------------------
# Original AA-CRC helpers are defined above.
# -----------------------------------------------------------------------------


def fit_one_risk_model(X: np.ndarray, z: np.ndarray, seed: int, backend: str = RISK_BACKEND):
    """Fit a single risk regressor at one lambda anchor.

    ``kv_cache`` is set in the TabICL constructor so that the in-context
    representation is built once and reused across the prediction batches.
    """
    if backend == "tabicl":
        return TabICLRegressor(
            n_estimators=TABICL_ESTIMATORS,
            device=TABICL_DEVICE,
            kv_cache=True,
            random_state=seed,
        ).fit(X, z)

    return HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=8,
        learning_rate=0.05,
        random_state=seed,
    ).fit(X, z)


def predict_batched(model, F: np.ndarray, batch_rows: int = BATCH_ROWS_TABICL) -> np.ndarray:
    """Predict in row batches to keep the TabICL memory footprint bounded."""
    n = F.shape[0]
    out = np.empty(n, np.float32)
    for s in range(0, n, batch_rows):
        out[s : s + batch_rows] = model.predict(F[s : s + batch_rows])
    return out


def predict_risk_table(
    models: Sequence, F: np.ndarray, lam_train: np.ndarray, lam_grid: np.ndarray
) -> np.ndarray:
    """Assemble the conditional risk surface R(x, lambda) over the full lambda grid.

    Predictions at the trained anchors are clipped to [0, 1] and made monotone
    non-increasing in lambda, then interpolated with a monotone PCHIP spline and
    made monotone again on the dense grid.
    """
    n = F.shape[0]
    p = np.empty((n, len(lam_train)), np.float32)
    for j in range(len(lam_train)):
        p[:, j] = predict_batched(models[j], F)
    p = np.minimum.accumulate(np.clip(p, 0, 1), 1)

    out = np.empty((n, len(lam_grid)), np.float32)
    for i in range(n):
        out[i] = np.clip(PchipInterpolator(lam_train, p[i], extrapolate=True)(lam_grid), 0, 1)
    return np.minimum.accumulate(out, 1)


def invert_risk(R: np.ndarray, a: float) -> np.ndarray:
    """Smallest lambda index whose estimated conditional risk is within budget a."""
    ok = R <= a
    return np.where(ok.any(1), np.argmax(ok, 1), R.shape[1] - 1)


try:
    from .risk_calibration import RiskCalibration
except ImportError:
    from risk_calibration import RiskCalibration


def run_recirc(
    F_D: np.ndarray,
    L_D: np.ndarray,
    F_C: np.ndarray,
    L_C: np.ndarray,
    F_T: np.ndarray,
    lam_grid: np.ndarray,
    alpha: float,
    seed: int,
    backend: str = RISK_BACKEND,
    return_risk: bool = False,
) -> Tuple[np.ndarray, float]:
    """Fit the ReCIRC risk surface on D and calibrate the risk budget on C.

    One risk model is fitted per lambda anchor. For the TabICL backend the
    context D is subsampled to N_D_TABICL rows, since in-context performance
    saturates early while attention cost is quadratic in the context length.

    Returns the selected lambda index per test point and the calibrated budget.
    With ``return_risk=True``, also returns the already-computed held-out risk
    surface used by the model-independent calibration diagnostic.
    """
    idx = np.linspace(0, len(lam_grid) - 1, N_LAM_TRAIN, dtype=int)
    lam_train = lam_grid[idx]

    if backend == "tabicl" and len(F_D) > N_D_TABICL:
        sub = np.random.default_rng(seed).choice(len(F_D), N_D_TABICL, replace=False)
        F_Dt, L_Dt = F_D[sub], L_D[sub]
    else:
        F_Dt, L_Dt = F_D, L_D

    models = [fit_one_risk_model(F_Dt, L_Dt[:, j], seed, backend) for j in idx]  # one model per lambda

    R_C = predict_risk_table(models, F_C, lam_train, lam_grid)
    R_T = predict_risk_table(models, F_T, lam_train, lam_grid)

    ar = np.arange(F_C.shape[0])
    a_grid = A_GRID
    risks = np.array([L_C[ar, invert_risk(R_C, a)].mean() for a in a_grid])

    nC = F_C.shape[0]
    bound = crc_upper_bound(risks, n=nC, B=B_LOSS)
    v = np.where(bound <= alpha)[0]
    a_hat = a_grid[v[-1]] if len(v) else a_grid[0]

    if return_risk:
        return invert_risk(R_T, a_hat), float(a_hat), R_T
    return invert_risk(R_T, a_hat), float(a_hat)


# -----------------------------------------------------------------------------
# Splits and slice-based evaluation
# -----------------------------------------------------------------------------


def make_three_way_split(
    n: int, seed: int, frac_D: float = FRAC_D, frac_C: float = FRAC_C
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random permutation split into D (risk/context), C (calibration) and T (test)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nD, nC = int(frac_D * n), int(frac_C * n)
    return perm[:nD], perm[nD : nD + nC], perm[nD + nC :]


def slice_stats(
    loss: np.ndarray, slices: Dict[str, np.ndarray], alpha: float = ALPHA, nmin: int = SLICE_NMIN
) -> Dict[str, float]:
    """Summarise the empirical risk across evaluation slices.

    Only slices with at least ``nmin`` test points are considered. The CVaR is
    the mean of the worst decile of slices (at least one slice).
    """
    r = np.array([loss[m].mean() for m in slices.values() if m.sum() >= nmin])
    k = max(1, len(r) // 10)
    return dict(
        worst=float(r.max()),
        cvar=float(np.sort(r)[-k:].mean()),
        excess=float(np.maximum(r - alpha, 0).mean()),
        frac=float((r > alpha).mean()),
    )


# -----------------------------------------------------------------------------
# Single trial loop
# -----------------------------------------------------------------------------


def run_one_trial(
    trial: int,
    seed: int,
    X: np.ndarray,
    Y: np.ndarray,
    slice_fn: Callable[[np.ndarray], Dict[str, np.ndarray]],
    alpha: float = ALPHA,
    backend: str = RISK_BACKEND,
    use_recirc: bool = True,
    diagnostic: Optional[RiskCalibration] = None,
    save_predictions: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Run the three arms on a single split and collect summary and slice rows."""
    n = len(X)
    idx_D, idx_C, idx_T = make_three_way_split(n, seed)

    (q_lo, q_med, q_hi), (q_lo_oob, q_med_oob, q_hi_oob) = fit_qrf(X, Y, idx_D, seed)
    LOSS, WIDTH, lam, s_neg, s_pos = build_loss(Y, q_lo, q_med, q_hi)
    F = feats(X, q_med, s_neg, s_pos)
    # OOB versions (only the D rows differ from the in-sample ones).
    LOSS_oob, _, _, s_neg_oob, s_pos_oob = build_loss(Y, q_lo_oob, q_med_oob, q_hi_oob)
    F_oob = feats(X, q_med_oob, s_neg_oob, s_pos_oob)

    L_C, L_T, W_T = LOSS[idx_C], LOSS[idx_T], WIDTH[idx_T]
    ar = np.arange(len(idx_T))

    sl = slice_fn(idx_T)

    outs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    selected: Dict[str, float] = {}
    own_groups: Dict[str, float] = {}

    saturation: Dict[str, float] = {}
    top = len(lam) - 1

    # Covariates given to both AA-CRC arms (raw X by default, or ReCIRC's feats).
    if AACRC_FEATURES == "feats":
        Z_D, Z_C, Z_T = F_oob[idx_D], F[idx_C], F[idx_T]
    else:
        Z_D, Z_C, Z_T = X[idx_D], X[idx_C], X[idx_T]
    aa_args = (Y[idx_C], q_med[idx_C], s_neg[idx_C], s_pos[idx_C])

    # --- Arm 1: marginal CRC, calibrated on C only ---------------------------
    j = crc_global(L_C, alpha)
    outs[METHOD_CRC] = (L_T[:, j], W_T[:, j])
    selected[METHOD_CRC] = float(lam[j])
    saturation[METHOD_CRC] = float(j == top)

    def diag_file(tag):
        if AACRC_OUTPUT_DIR is None:
            return None
        return AACRC_OUTPUT_DIR / f"aacrc_{tag}_diagnostics_trial_{trial:03d}.json"

    def deploy(method, tag, u, truncate):
        loss_aa, width_aa, lam_aa, diag_lam = evaluate_aacrc_intervals(
            Y[idx_T], q_med[idx_T], s_neg[idx_T], s_pos[idx_T], u,
            lam_grid=lam, truncate=truncate, snap_to_grid=AACRC_SNAP_TO_GRID,
        )
        outs[method] = (loss_aa, width_aa)
        selected[method] = float(np.mean(lam_aa))
        saturation[method] = float(diag_lam["rate_at_lam_max"])
        if save_predictions and AACRC_OUTPUT_DIR is not None:
            pd.DataFrame({"test_index": idx_T, "threshold_u": u, "lambda": lam_aa,
                          "loss": loss_aa, "width": width_aa}).to_csv(
                AACRC_OUTPUT_DIR / f"aacrc_{tag}_predictions_trial_{trial:03d}.csv", index=False
            )
        path = diag_file(tag)
        if path is not None and path.exists():
            payload = json.loads(path.read_text())
            payload["deployment_multiplier"] = diag_lam
            path.write_text(json.dumps(payload, indent=2, allow_nan=False))
        return loss_aa

    # --- Arm 2a: AA-CRC, linear feature map, theta on C ------------------------
    if AACRC_BASIS in ("linear", "both"):
        fit_lin = fit_original_aacrc(
            Z_D, Z_C, *aa_args, alpha=alpha, ridge=AACRC_RIDGE, maxiter=AACRC_MAXITER,
            module=AACRC_MODULE, diagnostic_path=diag_file("linear"),
            ridge_intercept=AACRC_RIDGE_INTERCEPT,
        )
        deploy(METHOD_AACRC_LIN, "linear", predict_original_aacrc(fit_lin, Z_T), AACRC_TRUNCATE)

    # --- Arm 2b: AA-CRC, RF-leaf feature map (Algorithm 1), theta on C -----------
    if AACRC_BASIS in ("rf", "both"):
        target_D = np.log1p(minimal_cover_multiplier(
            Y[idx_D], q_med_oob[idx_D], s_neg_oob[idx_D], s_pos_oob[idx_D]
        ))
        leaf_map = LeafFeatureMap(Z_D, target_D, n_trees=RF_TREES, max_depth=RF_DEPTH,
                                  min_leaf=RF_MIN_LEAF, seed=seed)
        fit_rf = fit_aacrc_rf(
            leaf_map, Z_C, *aa_args, alpha=alpha, lam_init=float(lam[j]), lam_max=LAM_MAX,
            maxiter=AACRC_MAXITER, module=AACRC_MODULE, diagnostic_path=diag_file("rf"),
        )
        phi_T = leaf_map(Z_T)
        loss_rf = deploy(METHOD_AACRC_RF, "rf", phi_T @ fit_rf["theta"], truncate=True)
        own_groups[METHOD_AACRC_RF] = own_group_worst(loss_rf, phi_T)

    # --- Arm 3: ReCIRC, risk model on D, budget calibrated on C --------------
    if use_recirc:
        F_ctx, L_ctx = (F_oob, LOSS_oob) if RECIRC_OOB_CONTEXT else (F, LOSS)
        recirc_result = run_recirc(
            F_ctx[idx_D], L_ctx[idx_D], F[idx_C], L_C, F[idx_T], lam,
            alpha, seed, backend, return_risk=diagnostic is not None,
        )
        idx_rc, a_hat = recirc_result[:2]
        if diagnostic is not None:
            losses_by_budget = np.column_stack(
                [L_T[ar, invert_risk(recirc_result[2], a)] for a in A_GRID]
            )
            diagnostic.add_trial(
                A_GRID, losses_by_budget, sl, trial=trial, seed=seed, a_hat=a_hat
            )
        outs[METHOD_RECIRC] = (L_T[ar, idx_rc], W_T[ar, idx_rc])
        selected[METHOD_RECIRC] = float(a_hat)
        saturation[METHOD_RECIRC] = float((idx_rc == top).mean())

    summary_rows: List[Dict] = []
    slice_rows: List[Dict] = []

    for method, (loss, width) in outs.items():
        st = slice_stats(loss, sl, alpha)
        summary_rows.append(
            {
                "method": method,
                "trial": trial,
                "seed": seed,
                "alpha": alpha,
                "risk_backend": backend if use_recirc else None,
                "selected_param": selected[method],
                "marginal_risk": float(loss.mean()),
                "excess_risk_event": float(loss.mean() > alpha),
                "avg_width": float(width.mean()),
                "median_width": float(np.median(width)),
                "infinite_width_rate": float(np.isinf(width).mean()),
                "rate_at_lam_max": float(saturation.get(method, float("nan"))),
                "own_group_worst": float(own_groups.get(method, float("nan"))),
                "worst_slice": st["worst"],
                "slice_cvar": st["cvar"],
                "mean_excess_slice": st["excess"],
                "frac_slices_above_alpha": st["frac"],
                "n_test": int(len(idx_T)),
            }
        )

        for name, msk in sl.items():
            if msk.sum() >= SLICE_NMIN:
                slice_rows.append(
                    {
                        "method": method,
                        "trial": trial,
                        "slice": name,
                        "n": int(msk.sum()),
                        "slice_risk": float(loss[msk].mean()),
                        "mean_width": float(width[msk].mean()),
                    }
                )

    return summary_rows, slice_rows


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def width_mean(values):
    """Preserve infinite widths in aggregation across pandas versions."""
    return float(np.asarray(values, dtype=float).mean())


def width_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 and np.isfinite(values).all() else float("nan")


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-trial metrics by method."""
    if len(df) == 0:
        return pd.DataFrame()
    return (
        df.groupby("method", as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            marginal_risk_mean=("marginal_risk", "mean"),
            marginal_risk_sd=("marginal_risk", "std"),
            excess_event_rate=("excess_risk_event", "mean"),
            avg_width_mean=("avg_width", width_mean),
            avg_width_sd=("avg_width", width_sd),
            infinite_width_rate=("infinite_width_rate", "mean"),
            rate_at_lam_max=("rate_at_lam_max", "mean"),
            own_group_worst_mean=("own_group_worst", "mean"),
            worst_slice_mean=("worst_slice", "mean"),
            worst_slice_sd=("worst_slice", "std"),
            slice_cvar_mean=("slice_cvar", "mean"),
            mean_excess_slice=("mean_excess_slice", "mean"),
            frac_slices_above_alpha=("frac_slices_above_alpha", "mean"),
        )
        .sort_values("worst_slice_mean")
    )


def aggregate_slices(df_slices: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-slice risk and width by method and slice."""
    if len(df_slices) == 0:
        return pd.DataFrame()
    return df_slices.groupby(["method", "slice"], as_index=False).agg(
        slice_risk_mean=("slice_risk", "mean"),
        slice_risk_sd=("slice_risk", "std"),
        mean_width=("mean_width", width_mean),
        n_mean=("n", "mean"),
    )


def paired_comparison(df: pd.DataFrame, m1: str, m2: str, metric: str = "worst_slice") -> Optional[Dict]:
    """Paired difference m1 - m2 across trials (negative means m1 is better)."""
    present = df["method"].unique()
    if m1 not in present or m2 not in present:
        return None
    a = df[df.method == m1].sort_values("trial")[metric].to_numpy()
    b = df[df.method == m2].sort_values("trial")[metric].to_numpy()
    if len(a) != len(b) or len(a) == 0:
        return None
    d = a - b
    return {
        "comparison": f"{m1} − {m2}",
        "metric": metric,
        "n_trials": int(len(d)),
        "mean_diff": float(d.mean()),
        "sd_diff": float(d.std(ddof=1)) if len(d) > 1 and np.isfinite(d).all() else float("nan"),
        "win_rate_first": float((d < 0).mean()),
    }


def build_paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the paired comparisons reported in the notebook, plus width."""
    rows = []
    for metric in ("worst_slice", "slice_cvar", "marginal_risk", "avg_width"):
        for m1, m2 in ((METHOD_RECIRC, METHOD_AACRC_RF), (METHOD_RECIRC, METHOD_AACRC_LIN),
                       (METHOD_RECIRC, METHOD_CRC), (METHOD_AACRC_RF, METHOD_AACRC_LIN),
                       (METHOD_AACRC_RF, METHOD_CRC), (METHOD_AACRC_LIN, METHOD_CRC)):
            r = paired_comparison(df, m1, m2, metric)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def build_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Paper-ready compact table: mean ± sd for the headline metrics."""
    if len(summary) == 0:
        return pd.DataFrame()
    out = summary.copy()

    def fmt(mean_col: str, sd_col: str, digits: int) -> pd.Series:
        return out.apply(
            lambda r: f"{r[mean_col]:.{digits}f} ± {r[sd_col]:.{digits}f}",
            axis=1,
        )

    out["marginal_risk"] = fmt("marginal_risk_mean", "marginal_risk_sd", 4)
    out["worst_slice_risk"] = fmt("worst_slice_mean", "worst_slice_sd", 4)
    out["avg_width"] = fmt("avg_width_mean", "avg_width_sd", 3)
    return out[
        [
            "method",
            "n_trials",
            "marginal_risk",
            "avg_width",
            "infinite_width_rate",
            "worst_slice_risk",
            "slice_cvar_mean",
            "mean_excess_slice",
        ]
    ]


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def save_plots(
    df: pd.DataFrame,
    df_slices: pd.DataFrame,
    output_dir: str,
    slice_features: Sequence[str],
    alpha: float = ALPHA,
) -> None:
    """Save the method-level bar chart, the per-trial scatter and the slice heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    methods = [m for m in METHODS_ORDER if m in df["method"].unique()]
    if not methods:
        return

    # --- Figure 1: worst-slice risk and average width by method --------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (col, title, show_alpha) in enumerate(
        [("worst_slice", "Worst-slice risk", True), ("avg_width", "Average width (K)", False)]
    ):
        means = [df[df.method == m][col].mean() for m in methods]
        stds = [df[df.method == m][col].std() if len(df[df.method == m]) > 1 else 0.0 for m in methods]
        x = np.arange(len(methods))
        axes[j].bar(x, means, yerr=stds, color=[PALETTE.get(m, "gray") for m in methods], capsize=4, alpha=0.85)
        if show_alpha:
            axes[j].axhline(alpha, ls="--", c="#c00", lw=1, label=fr"$\alpha={alpha}$")
            axes[j].legend(fontsize=8)
        axes[j].set_xticks(x)
        axes[j].set_xticklabels(methods, rotation=15, ha="right")
        axes[j].set_title(title)
        axes[j].grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"Superconductor | alpha={alpha} | {df['trial'].nunique()} splits",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "worst_slice_and_width_by_method.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-trial dispersion of the four headline metrics ---------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    panels = [
        ("marginal_risk", r"$\widehat{R}_{\mathrm{test}}$", "(a) Marginal risk", True),
        ("worst_slice", r"$\max_s \widehat{R}_s$", "(b) Worst slice (top-2 feature bins)", True),
        ("mean_excess_slice", "Mean excess", "(c) Mean slice excess", False),
        ("avg_width", "Interval width (K)", "(d) Mean interval width", False),
    ]
    jit = np.random.default_rng(0)
    for ax, (metric, ylab, title, show_alpha) in zip(axes, panels):
        for j, m in enumerate(methods):
            tmp = df[df["method"] == m]
            x = np.full(len(tmp), j, float) + jit.normal(0, 0.04, len(tmp))
            ax.scatter(x, tmp[metric], alpha=0.75, s=45, color=PALETTE.get(m, "gray"))
            ax.hlines(tmp[metric].mean(), j - 0.2, j + 0.2, lw=2.5, color="black")
        if show_alpha:
            ax.axhline(alpha, ls="--", color="gray", lw=1.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9, rotation=15)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "marginal_metrics_by_method.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: slice risk heatmaps over the 4x4 bin interactions ---------
    agg = aggregate_slices(df_slices)
    if len(agg) == 0:
        return

    parsed = agg["slice"].str.split("_", expand=True).astype(int)
    agg = agg.assign(bin1=parsed[0], bin2=parsed[1])
    n_bins = int(max(agg["bin1"].max(), agg["bin2"].max())) + 1
    vmax = float(agg["slice_risk_mean"].max())

    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8), squeeze=False)
    for j, m in enumerate(methods):
        grid = np.full((n_bins, n_bins), np.nan)
        tmp = agg[agg["method"] == m]
        for _, r in tmp.iterrows():
            grid[int(r["bin1"]), int(r["bin2"])] = r["slice_risk_mean"]
        ax = axes[0, j]
        im = ax.imshow(grid, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(m, fontsize=10)
        ax.set_xlabel(f"Quantile bin: {slice_features[1] if len(slice_features) > 1 else 'feature 2'}", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"Quantile bin: {slice_features[0] if slice_features else 'feature 1'}", fontsize=8)
        ax.set_xticks(range(n_bins))
        ax.set_yticks(range(n_bins))
        fig.colorbar(im, ax=ax, fraction=0.046, label="Slice risk" if j == len(methods) - 1 else None)
    fig.suptitle(f"Slice risk across feature-bin interactions (alpha={alpha})", fontsize=12, y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "slice_risk_heatmap.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimento de mecanismo Superconductor: CRC vs AA-CRC (objetivo oficial) vs ReCIRC."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Diretório para salvar resultados.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Número de trials do experimento.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Seed base para randomização dos splits.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Nível de risco alvo.")
    parser.add_argument(
        "--slice-bins", type=int, default=N_SLICE_BINS, help="Bins por feature na definição dos slices."
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default=RISK_BACKEND,
        choices=["tabicl", "histgb"],
        help="Backbone do regressor de risco do ReCIRC.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Desabilita geração de gráficos.")
    parser.add_argument("--save-predictions", action="store_true", help="Salva CSVs AA-CRC por observação/trial.")
    parser.add_argument("--aacrc-repo", type=Path, default=AACRC_REPO)
    parser.add_argument("--aacrc-integration", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--aacrc-ridge", type=float, default=AACRC_RIDGE,
                        help="Ridge do braço linear (só inclinações, salvo --aacrc-ridge-intercept).")
    parser.add_argument("--aacrc-basis", choices=["linear", "rf", "both"], default="both",
                        help="Feature map(s) do AA-CRC: linear, folhas de RF (Algoritmo 1) ou ambos.")
    parser.add_argument("--aacrc-features", choices=["x", "feats"], default="x",
                        help="Covariáveis do AA-CRC: X original (autores) ou as mesmas features do ReCIRC.")
    parser.add_argument("--aacrc-ridge-intercept", action="store_true",
                        help="Penaliza também o intercepto no braço linear (ridge original dos autores).")
    parser.add_argument("--rf-trees", type=int, default=RF_TREES)
    parser.add_argument("--rf-depth", type=int, default=RF_DEPTH)
    parser.add_argument("--rf-min-leaf", type=int, default=RF_MIN_LEAF)
    parser.add_argument("--recirc-oob-context", action="store_true",
                        help="Perdas de contexto do ReCIRC a partir de quantis OOB da QRF em D.")
    parser.add_argument("--aacrc-maxiter", type=int, default=AACRC_MAXITER)
    parser.add_argument(
        "--no-aacrc-truncate", action="store_true",
        help="Nao trunca o multiplicador do AA-CRC linear em LAM_MAX (intervalos infinitos "
             "quando u <= 0). O braço RF nunca precisa de truncamento.",
    )
    parser.add_argument(
        "--aacrc-snap-to-grid", action="store_true",
        help="Arredonda o multiplicador do AA-CRC para a grade de N_LAM pontos usada por CRC/ReCIRC.",
    )
    parser.add_argument("--data-file", type=Path, default=Path(__file__).resolve().parent / "data_superconductor/train.csv")
    parser.add_argument("--no-recirc", action="store_true", help="Executa apenas CRC e AA-CRC.")
    parser.add_argument("--self-check", action="store_true", help="Validação offline com NumPy/SciPy.")
    args = parser.parse_args()
    if args.trials < 1 or not 0 < args.alpha < 1 or args.slice_bins < 1:
        parser.error("--trials e --slice-bins devem ser positivos; --alpha deve estar entre 0 e 1.")
    if not np.isfinite(args.aacrc_ridge) or args.aacrc_ridge < 0 or args.aacrc_maxiter < 1:
        parser.error("--aacrc-ridge deve ser finito e não negativo; --aacrc-maxiter deve ser positivo.")
    if min(args.rf_trees, args.rf_depth, args.rf_min_leaf) < 1:
        parser.error("--rf-trees, --rf-depth e --rf-min-leaf devem ser positivos.")
    return args


def main():
    global AACRC_MODULE, AACRC_OUTPUT_DIR, AACRC_RIDGE, AACRC_MAXITER, AACRC_INTEGRATION
    global AACRC_TRUNCATE, AACRC_SNAP_TO_GRID, METHODS_ORDER
    global AACRC_BASIS, AACRC_FEATURES, AACRC_RIDGE_INTERCEPT, RECIRC_OOB_CONTEXT
    global RF_TREES, RF_DEPTH, RF_MIN_LEAF
    args = parse_args()

    alpha = args.alpha
    n_trials = args.trials
    base_seed = args.seed
    backend = args.risk_model

    output_dir = Path(args.output_dir) if args.output_dir else Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_recirc = not args.no_recirc
    if use_recirc and backend == "tabicl" and not HAS_TABICL:
        raise RuntimeError("TabICL indisponível. Instale tabicl, selecione --risk-model histgb ou --no-recirc.")
    AACRC_INTEGRATION = args.aacrc_integration
    AACRC_MODULE = load_original_aacrc(args.aacrc_repo, AACRC_INTEGRATION)
    AACRC_OUTPUT_DIR = output_dir
    AACRC_RIDGE, AACRC_MAXITER = args.aacrc_ridge, args.aacrc_maxiter
    AACRC_TRUNCATE = not args.no_aacrc_truncate
    AACRC_SNAP_TO_GRID = bool(args.aacrc_snap_to_grid)
    AACRC_BASIS, AACRC_FEATURES = args.aacrc_basis, args.aacrc_features
    AACRC_RIDGE_INTERCEPT = bool(args.aacrc_ridge_intercept)
    RECIRC_OOB_CONTEXT = bool(args.recirc_oob_context)
    RF_TREES, RF_DEPTH, RF_MIN_LEAF = args.rf_trees, args.rf_depth, args.rf_min_leaf
    METHODS_ORDER = [METHOD_CRC]
    if AACRC_BASIS in ("linear", "both"):
        METHODS_ORDER.append(METHOD_AACRC_LIN)
    if AACRC_BASIS in ("rf", "both"):
        METHODS_ORDER.append(METHOD_AACRC_RF)
    if use_recirc:
        METHODS_ORDER.append(METHOD_RECIRC)
    if not AACRC_TRUNCATE:
        print('AVISO: multiplicador do AA-CRC nao truncado; larguras infinitas sao possiveis.')
    print(f"config OK | regressor de risco: {backend if use_recirc else 'disabled'} | device: {TABICL_DEVICE}")
    X, Y, names = load_superconductor(str(args.data_file))
    if int(FRAC_C * len(Y)) <= 1.0 / alpha:
        raise ValueError("Superconductor precisa de len(C) > 1/alpha para ajustar AA-CRC.")
    slice_fn, slice_features = build_slice_function(X, Y, names, n_bins=args.slice_bins)

    summary_rows: List[Dict] = []
    slice_rows: List[Dict] = []
    risk_diagnostic = RiskCalibration(ylabel="Average loss (exploratory slices)")

    t0 = time.time()
    for t in range(n_trials):
        seed = base_seed + t
        s_rows, sl_rows = run_one_trial(
            trial=t, seed=seed, X=X, Y=Y, slice_fn=slice_fn, alpha=alpha,
            backend=backend, use_recirc=use_recirc,
            diagnostic=risk_diagnostic if use_recirc else None,
            save_predictions=args.save_predictions,
        )
        summary_rows.extend(s_rows)
        slice_rows.extend(sl_rows)

        pd.DataFrame(summary_rows).to_csv(output_dir / "trial_results_incremental.csv", index=False)

        if (t + 1) % 5 == 0 or (t + 1) == n_trials:
            print(f"trial {t + 1}/{n_trials} | {time.time() - t0:.0f}s acumulados")

    risk_diagnostic.save(output_dir, make_plot=not args.no_plots)

    df = pd.DataFrame(summary_rows)
    df_slices = pd.DataFrame(slice_rows)
    elapsed = time.time() - t0

    df.to_csv(output_dir / "trial_results.csv", index=False)
    df_slices.to_csv(output_dir / "slice_results.csv", index=False)

    summary = aggregate_summary(df)
    slices_agg = aggregate_slices(df_slices)
    compact = build_compact_table(summary)
    paired = build_paired_table(df)

    summary.to_csv(output_dir / "summary_by_method.csv", index=False)
    slices_agg.to_csv(output_dir / "slices_by_method.csv", index=False)
    compact.to_csv(output_dir / "compact_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)

    print(f"\n=== Superconductor — AA-CRC ({AACRC_BASIS}) — {n_trials} trials | {elapsed:.0f}s ===\n")
    print(summary.round(4).to_string(index=False))
    print("\nTabela compacta:")
    print(compact.to_string(index=False))

    if len(paired):
        print("\nComparações pareadas (negativo = primeiro método melhor):")
        for _, r in paired.iterrows():
            print(
                f"  [{r['metric']}] {r['comparison']}: {r['mean_diff']:+.4f}±{r['sd_diff']:.4f} | "
                f"primeiro melhor em {r['win_rate_first'] * 100:.0f}% dos trials"
            )

    if not args.no_plots and not np.isfinite(df["avg_width"]).all():
        warnings.warn("Larguras infinitas preservadas nos CSVs; gráficos omitidos.")
    elif not args.no_plots:
        save_plots(df, df_slices, str(output_dir), slice_features=slice_features, alpha=alpha)
        print(f"\nGráficos salvos em: {output_dir}")

    meta = {
        "experiment": "mechanism_superconductor_official_aacrc_rf_leaves",
        "aacrc": {
            "source_commit": AACRC_COMMIT, "source_sha256": AACRC_SOURCE_SHA256,
            "source_path": str(Path(AACRC_MODULE.__file__).resolve()),
            "integration": AACRC_INTEGRATION, "ridge": AACRC_RIDGE, "maxiter": AACRC_MAXITER,
            "fit_split": "C", "basis": AACRC_BASIS, "covariates": AACRC_FEATURES,
            "linear_arm": {
                "feature_map": "intercept + covariates standardized with D statistics",
                "ridge": AACRC_RIDGE, "ridge_on_intercept": AACRC_RIDGE_INTERCEPT,
                "truncate_at_lam_max": bool(AACRC_TRUNCATE),
            },
            "rf_arm": {
                "feature_map": "one-hot RF leaf indicators (Blot et al., Algorithm 1)",
                "rf_fit_split": "D", "rf_target": "log1p(minimal covering multiplier), OOB QRF quantiles",
                "n_trees": RF_TREES, "max_depth": RF_DEPTH, "min_samples_leaf": RF_MIN_LEAF,
                "ridge": 0.0, "bounds": "theta_j in [exp(-LAM_MAX)/T, 1/T]",
            },
            "threshold_class": "u(x) = Phi(x) @ theta",
            "multiplier": "max(0, -log(u)) for u > 0; infinity otherwise",
            "loss_reduction": "five positive auxiliary labels, lower:upper multiplicity 1:4",
            "continuous_multiplier": True, "postfit_scalar_calibration": False,
            "theory_note": "task-specific adaptation of the authors' objective; no additional validity claim",
        },
        "data_file": str(args.data_file.resolve()),
        "recirc_enabled": use_recirc,
        "recirc_oob_context": RECIRC_OOB_CONTEXT,
        "slice_selection": "top-2 absolute target correlations on full data; exploratory",
        "alpha": float(alpha),
        "n_trials": int(n_trials),
        "seed": int(base_seed),
        "n_obs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "slice_features": list(slice_features),
        "slice_bins": int(args.slice_bins),
        "save_predictions": bool(args.save_predictions),
        "risk_backend": str(backend) if use_recirc else None,
        "tabicl_device": str(TABICL_DEVICE),
        "tabicl_context_size": int(N_D_TABICL),
        "frac_D": float(FRAC_D),
        "frac_C": float(FRAC_C),
        "loss_weights": {"w_neg": float(W_NEG), "w_pos": float(W_POS)},
        "quantiles": [float(Q_LO), float(Q_MED), float(Q_HI)],
        "lambda_grid": {"methods": ["CRC", "ReCIRC"], "n": int(N_LAM), "max": float(LAM_MAX), "n_train_anchors": int(N_LAM_TRAIN)},
        "loss_bound_B": float(B_LOSS),
        "aacrc_deployment": {"truncate_at_lam_max": bool(AACRC_TRUNCATE),
                             "snap_to_grid": bool(AACRC_SNAP_TO_GRID)},
        "budget_grid": {
            "construction": "fixed_before_calibration",
            "n": int(A_GRID.size),
            "min": float(A_GRID[0]),
            "max": float(A_GRID[-1]),
            "values": A_GRID.tolist(),
        },
        "qrf": {
            "n_estimators": int(N_TREES_QRF),
            "max_depth": int(MAX_DEPTH_QRF),
            "min_samples_leaf": int(MIN_LEAF_QRF),
        },
        "slice_nmin": int(SLICE_NMIN),
        "elapsed_seconds": float(elapsed),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
