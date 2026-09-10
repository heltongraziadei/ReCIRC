#!/usr/bin/env python
"""Medical Insurance: marginal CRC, truncated AA-CRC, and ReCIRC.

Uses J/J_prime from the pinned local AA-CRC repository, with SLSQP and the
exact asymmetric-loss encoding used in Experiment 7. One lower-tail label
and four upper-tail labels encode weights (0.2, 0.8). The affine learned
threshold is u(x); raw interval multipliers are max(0, -log(u)) for u > 0 and
infinity otherwise. This is a task adaptation of the original objective,
not an unchanged reproduction of the authors' regression experiment.

QRF and feature standardization use D; AA-CRC uses independent C, with sample
correction 1/len(C). No least-squares surrogate or scalar-offset calibration.
At prediction time AA-CRC multipliers are clipped to [0, LAM_MAX=4], including
infinite raw multipliers. They remain continuous. Fitting and loss-encoding
checks use the untruncated mapping. Truncation can increase miscoverage; its
frequency and risk increase are reported. CRC/ReCIRC retain the previous
finite lambda grid and ReCIRC's fixed 201-point budget grid on [0, 1]. Their
finite-grid fallback is inherited and is not a fully protective endpoint.

Results go to a separate directory. Original scripts and author source are
preserved. --self-check validates the encoding, objective, gradient and fit
using only NumPy/SciPy, including finite deployment and JSON diagnostics.
--no-recirc runs CRC and AA-CRC without TabICL. This variant adds no validity
guarantee for the intervals after truncation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
# Shared cap is available before the lightweight --self-check entry point.
LAM_MAX = 4.0


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


def multiplier_from_aacrc_threshold(u, lam_max=LAM_MAX):
    """Finite deployment multiplier; lam_max=None exposes the raw fit mapping."""
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all():
        raise ValueError("AA-CRC score thresholds must be finite.")
    lam = np.full(u.shape, np.inf)
    positive = u > 0
    lam[positive] = np.maximum(0.0, -np.log(u[positive]))
    if lam_max is not None:
        if not np.isfinite(lam_max) or lam_max <= 0:
            raise ValueError("lam_max must be finite and strictly positive.")
        lam = np.minimum(lam, lam_max)
    return lam


def evaluate_aacrc_intervals(y, median, s_neg, s_pos, u, lam_max=LAM_MAX):
    lam = multiplier_from_aacrc_threshold(u, lam_max=lam_max)
    lo = np.asarray(median, float) - lam * np.asarray(s_neg, float)
    hi = np.asarray(median, float) + lam * np.asarray(s_pos, float)
    loss = 0.2 * (np.asarray(y) < lo) + 0.8 * (np.asarray(y) > hi)
    return loss, hi - lo, lam


def aacrc_truncation_diagnostics(u, lam_max=LAM_MAX):
    """JSON-safe diagnostics, including when every raw interval is infinite."""
    raw = multiplier_from_aacrc_threshold(u, lam_max=None)
    deployed = multiplier_from_aacrc_threshold(u, lam_max=lam_max)
    finite_raw = raw[np.isfinite(raw)]
    return {
        "lam_max": float(lam_max),
        "rate_u_nonpositive": float(np.mean(np.asarray(u) <= 0)),
        "rate_truncated": float(np.mean(raw > lam_max)),
        "rate_at_lam_max": float(np.mean(deployed >= lam_max)),
        "raw_max_finite": float(finite_raw.max()) if finite_raw.size else None,
        "raw_median_finite": float(np.median(finite_raw)) if finite_raw.size else None,
    }


def fit_original_aacrc(X_D, X_C, y_C, median_C, s_neg_C, s_pos_C,
                       alpha=0.1, ridge=0.01, maxiter=200, module=None,
                       diagnostic_path=None):
    """Optimize the original regularized AA-CRC objective on independent C."""
    if module is None:
        module = load_original_aacrc(AACRC_REPO, AACRC_INTEGRATION)
    n = len(y_C)
    if not 0 < alpha < 1 or n <= 1.0 / alpha:
        raise ValueError("Need 0 < alpha < 1 and len(C) > 1/alpha for this AA-CRC fit.")
    if ridge < 0 or maxiter < 1:
        raise ValueError("ridge must be nonnegative and maxiter positive.")
    center = np.asarray(X_D, float).mean(axis=0)
    scale = np.asarray(X_D, float).std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    phi = np.column_stack([np.ones(n), (np.asarray(X_C, float) - center) / scale])
    labels, scores = auxiliary_tail_labels(y_C, median_C, s_neg_C, s_pos_C)
    theta = np.zeros(phi.shape[1])
    theta[0] = 0.5
    regularization = "ridge" if ridge > 0 else None
    arguments = (labels, scores, phi, alpha, n, regularization, ridge)
    attempts = []
    start = time.time()
    success = False
    for iterations in (maxiter, 3 * maxiter):
        result = minimize(module.J, theta, method="SLSQP", jac=module.J_prime,
                          args=arguments, tol=1e-8,
                          options={"maxiter": iterations, "disp": False})
        success = bool(result.success and np.isfinite(result.x).all() and np.isfinite(result.fun))
        attempts.append({"success": success, "status": int(result.status),
                         "message": str(result.message), "iterations": int(result.nit),
                         "objective": float(result.fun) if np.isfinite(result.fun) else None})
        if success:
            break
        if np.isfinite(result.x).all():
            theta = result.x.copy()
    diag = {"source_commit": AACRC_COMMIT, "source_sha256": AACRC_SOURCE_SHA256,
            "fit_split": "C", "feature_standardization_split": "D", "n_fit": n,
            "n_auxiliary_labels_per_observation": 5, "ridge": ridge,
            "success": success, "attempts": attempts, "elapsed_seconds": time.time() - start}
    if success:
        theta = result.x.copy()
        u = phi @ theta
        if np.any(u < -module.INF_BORN_INT):
            success = False
            diag["success"] = False
            diag["error"] = "Fitted thresholds reached the original objective's lower truncation."
        loss, _, _ = evaluate_aacrc_intervals(y_C, median_C, s_neg_C, s_pos_C, u, lam_max=None)
        deployed_loss, _, _ = evaluate_aacrc_intervals(y_C, median_C, s_neg_C, s_pos_C, u)
        encoded_loss = module._I_prime_list(labels, scores, np.maximum(u, 0.0), alpha, n) + alpha - 1.0 / n
        diag.update({"theta": theta.tolist(), "center": center.tolist(), "scale": scale.tolist(),
                     "fit_risk": float(loss.mean()),
                     "fit_risk_truncated": float(deployed_loss.mean()),
                     "fit_truncation": aacrc_truncation_diagnostics(u),
                     "encoding_max_abs_error": float(np.max(np.abs(loss - encoded_loss))),
                     "threshold_min": float(u.min()), "threshold_max": float(u.max())})
        if diag["encoding_max_abs_error"] > 1e-10:
            success = False
            diag["success"] = False
            diag["error"] = "Interval/auxiliary-label losses disagree (check numerical boundary ties)."
    if diagnostic_path is not None:
        Path(diagnostic_path).write_text(json.dumps(diag, indent=2, allow_nan=False))
    if not success:
        raise RuntimeError("Original AA-CRC optimization failed: " + json.dumps(diag))
    return {"theta": theta, "center": center, "scale": scale, "diagnostics": diag}


def predict_original_aacrc(fit, X):
    phi = np.column_stack([np.ones(len(X)), (np.asarray(X, float) - fit["center"]) / fit["scale"]])
    return phi @ fit["theta"]


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
        direct, widths, _ = evaluate_aacrc_intervals(y, median, s_neg, s_pos, u, lam_max=None)
        encoded = module._I_prime_list(labels, scores, np.maximum(0, u), alpha, n) + alpha - 1 / n
        err = float(np.max(np.abs(direct - encoded)))
        max_loss_error = max(max_loss_error, err)
        assert err < 1e-12, (threshold, err)
        assert np.all(widths >= 0)
        deployed_loss, deployed_width, deployed_lam = evaluate_aacrc_intervals(y, median, s_neg, s_pos, u)
        assert np.isfinite(deployed_width).all()
        assert np.all((deployed_lam >= 0) & (deployed_lam <= LAM_MAX))
        assert np.all(deployed_width <= widths)
        assert np.all(deployed_loss >= direct)
        json.dumps(aacrc_truncation_diagnostics(u), allow_nan=False)
    # A response outside the finite endpoint must be counted as a miss.
    edge_u = np.array([-0.1, 0.0, np.nextafter(0.0, 1.0), np.exp(-LAM_MAX), 0.5, 1.0, 2.0])
    expected_lam = np.array([4.0, 4.0, 4.0, 4.0, np.log(2.0), 0.0, 0.0])
    assert np.allclose(multiplier_from_aacrc_threshold(edge_u), expected_lam)
    edge_loss, edge_width, _ = evaluate_aacrc_intervals(
        np.full(2, 25.0), np.full(2, 10.0), np.full(2, 2.0), np.full(2, 3.0), edge_u[:2]
    )
    assert np.all(edge_loss == 0.8) and np.all(edge_width == 20.0)
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
    direct, _, _ = evaluate_aacrc_intervals(y, median, s_neg, s_pos, u, lam_max=None)
    expected_gradient = np.mean(phi * (direct - target)[:, None], axis=0) + 2 * ridge * theta
    gradient = module.J_prime(theta, labels, scores, phi, alpha, n, "ridge", ridge)
    assert np.allclose(gradient, expected_gradient, atol=1e-12)
    fit = fit_original_aacrc(X_D, X_C, y, median, s_neg, s_pos, alpha=alpha, module=module)
    test_u = predict_original_aacrc(fit, X_C[:8])
    assert test_u.shape == (8,) and np.isfinite(test_u).all()
    print(json.dumps({"loss_identity_max_error": max_loss_error,
                      "official_quadrature_abs_error": quadrature_error,
                      "gradient_check": "passed", "truncation_check": "passed",
                      "optimizer": fit["diagnostics"]}, indent=2, allow_nan=False))


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

# Asymmetric loss weights: under-coverage below the interval is penalised less
# than above it (heavy right tail of medical charges).
W_NEG, W_POS = 0.2, 0.8

# QRF quantile levels used to build the (lower, median, upper) scaffold.
Q_LO, Q_MED, Q_HI = 0.05, 0.50, 0.95

# Lambda grid used by CRC and ReCIRC (multiplier applied to the asymmetric scales).
N_LAM = 80
EPS_SCALE = 1e-3

# Number of lambda anchors on which ReCIRC actually fits a risk regressor;
# the remaining grid points are recovered by monotone PCHIP interpolation.
N_LAM_TRAIN = 16

# Pre-calibration budget grid, fixed for every split (loss bound B = 1).
# Calibration losses select a budget from this grid; they never define it.
A_GRID = np.linspace(0.0, 1.0, 201)
A_GRID.setflags(write=False)

# Split fractions: D = risk/context, C = calibration, T = test (the remainder).
FRAC_D, FRAC_C = 0.40, 0.30

# QRF hyperparameters tuned for the small-sample regime (n_D ~ 535).
N_TREES_QRF = 200
MAX_DEPTH_QRF = 12
MIN_LEAF_QRF = 20

# Minimum slice size for a slice to enter the worst-slice statistic (T ~ 402).
SLICE_NMIN = 30

N_BINS = 5

RISK_MODEL = "tabicl"  # "tabicl" or "hgb"
TABICL_ESTIMATORS = 4
BATCH_ROWS = 4000

try:
    import torch

    TABICL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    TABICL_DEVICE = "cpu"

# Detect Google Drive mounted (Colab) and use as default
if os.path.isdir("/content/drive"):
    OUT_DIR = "/content/drive/MyDrive/PythonReCIRC/results/experiment_6_insurance_official_aacrc2_truncated_fixed_budget_grid"
else:
    OUT_DIR = "insurance_official_aacrc2_truncated_recirc_results_fixed_budget_grid"

# Method labels (kept stable across CSV outputs and figures).
METHOD_CRC = "CRC marginal"
METHOD_AACRC = "AA-CRC (official objective, truncated)"
METHOD_RECIRC_TABICL = "ReCIRC-TabICL"
METHOD_RECIRC_HGB = "ReCIRC-HGB"

PALETTE = {
    METHOD_CRC: "#1f77b4",
    METHOD_AACRC: "#ff7f0e",
    METHOD_RECIRC_TABICL: "#d62728",
    METHOD_RECIRC_HGB: "#2ca02c",
}


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

INSURANCE_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
)


@dataclass
class InsuranceData:
    """Container for the design matrix, target and the slice indicator masks.

    Attributes:
        X: Design matrix (n, p) with one-hot encoded categorical variables.
        Y: Annual medical charges (n,).
        smoker: Boolean mask, True for smokers.
        obese: Boolean mask, True when BMI >= 30.
        old: Boolean mask, True when age is above the sample median.
        raw: The original dataframe, kept for reporting.
    """

    X: np.ndarray
    Y: np.ndarray
    smoker: np.ndarray
    obese: np.ndarray
    old: np.ndarray
    raw: pd.DataFrame

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]


def load_insurance(cache_file: str = "./data_insurance/insurance.csv") -> InsuranceData:
    """Download (once) and preprocess the Medical Insurance dataset."""
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(cache_file):
        print("Downloading Medical Insurance dataset...")
        urllib.request.urlretrieve(INSURANCE_URL, cache_file)

    df = pd.read_csv(cache_file)
    expected = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no insurance.csv: {sorted(missing)}")

    Y = df["charges"].to_numpy(np.float32)
    Xdf = pd.get_dummies(df.drop(columns=["charges"]), columns=["sex", "smoker", "region"])
    for c in Xdf.columns:
        if Xdf[c].dtype == bool:
            Xdf[c] = Xdf[c].astype(np.int8)
    X = Xdf.to_numpy(np.float32)

    smoker = (df["smoker"] == "yes").to_numpy()
    obese = (df["bmi"] >= 30).to_numpy()
    old = (df["age"] >= df["age"].median()).to_numpy()

    return InsuranceData(X=X, Y=Y, smoker=smoker, obese=obese, old=old, raw=df)


def describe_interaction(data: InsuranceData) -> pd.DataFrame:
    """Tabulate mean/sd of charges across the smoker x obese cells."""
    rows = []
    for smk in [False, True]:
        for ob in [False, True]:
            s = data.Y[(data.smoker == smk) & (data.obese == ob)]
            rows.append(
                {
                    "smoker": bool(smk),
                    "bmi_ge_30": bool(ob),
                    "n": int(len(s)),
                    "mean_charges": float(s.mean()) if len(s) else np.nan,
                    "sd_charges": float(s.std()) if len(s) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Base predictor, loss construction and features
# -----------------------------------------------------------------------------


def fit_qrf(data: InsuranceData, idx_D: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the QRF scaffold on D and predict the (lo, med, hi) quantiles for all rows.

    The lower and upper quantiles are clipped against the median so that the
    resulting asymmetric scales are non-negative by construction.
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
    ).fit(data.X[idx_D], data.Y[idx_D])

    Q = qrf.predict(data.X, quantiles=[Q_LO, Q_MED, Q_HI])
    lo = np.minimum(Q[:, 0], Q[:, 1]).astype(np.float32)
    hi = np.maximum(Q[:, 2], Q[:, 1]).astype(np.float32)
    return lo, Q[:, 1].astype(np.float32), hi


def build_loss(
    data: InsuranceData, q_lo: np.ndarray, q_med: np.ndarray, q_hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the loss and width matrices over the lambda grid.

    Returns:
        LOSS: (n, N_LAM) asymmetric miscoverage loss for each row and lambda.
        WIDTH: (n, N_LAM) interval width in dollars.
        lam: the lambda grid.
        s_neg, s_pos: the lower/upper asymmetric scales.
    """
    s_neg = np.maximum(q_med - q_lo, EPS_SCALE).astype(np.float32)
    s_pos = np.maximum(q_hi - q_med, EPS_SCALE).astype(np.float32)
    lam = np.linspace(0, LAM_MAX, N_LAM).astype(np.float32)

    lo_b = q_med[:, None] - lam[None, :] * s_neg[:, None]
    hi_b = q_med[:, None] + lam[None, :] * s_pos[:, None]

    LOSS = (W_NEG * (data.Y[:, None] < lo_b) + W_POS * (data.Y[:, None] > hi_b)).astype(np.float32)
    WIDTH = (hi_b - lo_b).astype(np.float32)
    return LOSS, WIDTH, lam, s_neg, s_pos


def feats(data: InsuranceData, q_med: np.ndarray, s_neg: np.ndarray, s_pos: np.ndarray) -> np.ndarray:
    """Feature map used by the ReCIRC risk regressor: raw covariates + scale summaries."""
    w0 = s_neg + s_pos
    return np.column_stack(
        [
            data.X,
            q_med,
            s_neg,
            s_pos,
            w0,
            s_pos / (s_neg + s_pos),
            np.log1p(w0),
        ]
    ).astype(np.float32)


# -----------------------------------------------------------------------------
# CRC calibration
# -----------------------------------------------------------------------------


def crc_upper_bound(mean_loss: np.ndarray, n: int, B: float = 1.0) -> np.ndarray:
    """Finite-sample CRC upper bound (n / (n + 1)) * Rhat + B / (n + 1)."""
    return (n / (n + 1.0)) * mean_loss + B / (n + 1.0)


def crc_global(loss_cal: np.ndarray, alpha: float = ALPHA, B: float = 1.0) -> int:
    """Marginal CRC: smallest lambda index whose CRC bound is below alpha."""
    n = loss_cal.shape[0]
    rhat = loss_cal.mean(0)
    bound = crc_upper_bound(rhat, n=n, B=B)
    v = np.where(bound <= alpha)[0]
    return int(v[0]) if len(v) else loss_cal.shape[1] - 1


# -----------------------------------------------------------------------------
# AA-CRC with the authors' objective, fitted on C
# Original AA-CRC helpers are defined above.


def fit_one_risk_model(Xr: np.ndarray, z: np.ndarray, seed: int, backend: str):
    """Fit a single risk regressor at one lambda anchor.

    With n_D ~ 535, the whole context fits inside TabICL — no subsampling is
    needed, which is exactly the documented few-shot regime for this model.
    """
    if backend == "tabicl":
        return TabICLRegressor(
            n_estimators=TABICL_ESTIMATORS,
            device=TABICL_DEVICE,
            kv_cache=True,
            random_state=seed,
        ).fit(Xr, z)

    return HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=seed,
    ).fit(Xr, z)


def predict_batched(model, F: np.ndarray, batch_rows: int = BATCH_ROWS) -> np.ndarray:
    """Predict in row batches to keep the TabICL memory footprint bounded."""
    n = F.shape[0]
    out = np.empty(n, np.float32)
    for s in range(0, n, batch_rows):
        out[s : s + batch_rows] = model.predict(F[s : s + batch_rows])
    return out


def predict_risk_table(
    models: Sequence, F: np.ndarray, lam_train: np.ndarray, lam_grid: np.ndarray
) -> np.ndarray:
    """Assemble the conditional risk surface R(lambda | x) over the full lambda grid.

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


def run_recirc(
    F_D: np.ndarray,
    L_D: np.ndarray,
    F_C: np.ndarray,
    L_C: np.ndarray,
    F_T: np.ndarray,
    lam_grid: np.ndarray,
    alpha: float,
    seed: int,
    backend: str,
) -> Tuple[np.ndarray, float]:
    """Fit the ReCIRC risk surface on D and calibrate the risk budget on C.

    Returns the selected lambda index per test point and the calibrated budget.
    """
    idx = np.linspace(0, len(lam_grid) - 1, N_LAM_TRAIN, dtype=int)
    lam_train = lam_grid[idx]
    models = [fit_one_risk_model(F_D, L_D[:, j], seed, backend) for j in idx]

    R_C = predict_risk_table(models, F_C, lam_train, lam_grid)
    R_T = predict_risk_table(models, F_T, lam_train, lam_grid)

    ar = np.arange(F_C.shape[0])
    a_grid = A_GRID
    risks = np.array([L_C[ar, invert_risk(R_C, a)].mean() for a in a_grid])

    nC = F_C.shape[0]
    bound = crc_upper_bound(risks, n=nC, B=1.0)
    v = np.where(bound <= alpha)[0]
    a_hat = a_grid[v[-1]] if len(v) else a_grid[0]

    return invert_risk(R_T, a_hat), float(a_hat)


# -----------------------------------------------------------------------------
# Splits and evaluation by slices / difficulty bins
# -----------------------------------------------------------------------------


def make_three_way_split(
    n: int, seed: int, frac_D: float = FRAC_D, frac_C: float = FRAC_C
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random permutation split into D (risk/context), C (calibration) and T (test)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nD, nC = int(frac_D * n), int(frac_C * n)
    return perm[:nD], perm[nD : nD + nC], perm[nD + nC :]


def equal_mass_bins(v: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Label-free difficulty bins: quantile edges of the total scale s_neg + s_pos."""
    e = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return np.clip(np.digitize(v, e[1:-1]), 0, n_bins - 1)


def slice_masks(data: InsuranceData, idx_T: np.ndarray) -> Dict[str, np.ndarray]:
    """Interaction slices on the test split: smoker x obese and smoker x age."""
    sm, ob, ol = data.smoker[idx_T], data.obese[idx_T], data.old[idx_T]
    masks: Dict[str, np.ndarray] = {}
    for a_ in [0, 1]:
        for b_ in [0, 1]:
            masks[f"smk{a_}_ob{b_}"] = (sm == bool(a_)) & (ob == bool(b_))
            masks[f"smk{a_}_old{b_}"] = (sm == bool(a_)) & (ol == bool(b_))
    return masks


def slice_risks(
    loss: np.ndarray, data: InsuranceData, idx_T: np.ndarray, slice_nmin: int = SLICE_NMIN
) -> np.ndarray:
    """Per-slice empirical risk, restricted to slices with at least slice_nmin points."""
    masks = slice_masks(data, idx_T)
    return np.array([loss[m].mean() for m in masks.values() if m.sum() >= slice_nmin])


# -----------------------------------------------------------------------------
# Single trial loop
# -----------------------------------------------------------------------------


def run_one_trial(
    trial: int,
    seed: int,
    data: InsuranceData,
    alpha: float = ALPHA,
    risk_model: str = RISK_MODEL,
    use_recirc: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict], float]:
    """Run the three arms on a single split and collect marginal/conditional rows.

    Returns:
        marginal_rows, conditional_rows, slice_rows, crc_train_test_gap
    """
    idx_D, idx_C, idx_T = make_three_way_split(data.n, seed)

    q_lo, q_med, q_hi = fit_qrf(data, idx_D, seed)
    LOSS, WIDTH, lam, s_neg, s_pos = build_loss(data, q_lo, q_med, q_hi)
    F = feats(data, q_med, s_neg, s_pos)

    L_C, L_T, W_T = LOSS[idx_C], LOSS[idx_T], WIDTH[idx_T]
    ar = np.arange(len(idx_T))

    difficulty_T = (s_neg + s_pos)[idx_T]
    bins_T = equal_mass_bins(difficulty_T, N_BINS)

    outs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    selected: Dict[str, float] = {}

    # --- Arm 1: marginal CRC -------------------------------------------------
    j = crc_global(L_C, alpha)
    crc_gap = float(LOSS[idx_D][:, j].mean() - L_T[:, j].mean())
    outs[METHOD_CRC] = (L_T[:, j], W_T[:, j])
    selected[METHOD_CRC] = float(lam[j])

    # --- Arm 2: authors' AA-CRC objective fitted on independent C ------------
    diagnostic_path = (AACRC_OUTPUT_DIR / f"aacrc_diagnostics_trial_{trial:03d}.json"
                       if AACRC_OUTPUT_DIR is not None else None)
    fit = fit_original_aacrc(
        data.X[idx_D], data.X[idx_C], data.Y[idx_C], q_med[idx_C], s_neg[idx_C], s_pos[idx_C],
        alpha=alpha, ridge=AACRC_RIDGE, maxiter=AACRC_MAXITER,
        module=AACRC_MODULE, diagnostic_path=diagnostic_path,
    )
    u_aa = predict_original_aacrc(fit, data.X[idx_T])
    loss_aa, width_aa, lam_aa = evaluate_aacrc_intervals(
        data.Y[idx_T], q_med[idx_T], s_neg[idx_T], s_pos[idx_T], u_aa, lam_max=float(lam[-1])
    )
    raw_loss_aa, _, raw_lam_aa = evaluate_aacrc_intervals(
        data.Y[idx_T], q_med[idx_T], s_neg[idx_T], s_pos[idx_T], u_aa, lam_max=None
    )
    deployment_diag = aacrc_truncation_diagnostics(u_aa, lam_max=float(lam[-1]))
    deployment_diag.update({
        "raw_risk": float(raw_loss_aa.mean()),
        "truncated_risk": float(loss_aa.mean()),
        "risk_increase": float(np.mean(loss_aa - raw_loss_aa)),
    })
    fit["diagnostics"]["deployment"] = deployment_diag
    outs[METHOD_AACRC] = (loss_aa, width_aa)
    selected[METHOD_AACRC] = float(np.mean(lam_aa))
    if AACRC_OUTPUT_DIR is not None:
        diagnostic_path.write_text(json.dumps(fit["diagnostics"], indent=2, allow_nan=False))
        pd.DataFrame({"test_index": idx_T, "threshold_u": u_aa, "lambda": lam_aa,
                      "loss": loss_aa, "width": width_aa,
                      "was_truncated": raw_lam_aa > lam[-1],
                      "raw_would_be_infinite": u_aa <= 0,
                      "raw_loss": raw_loss_aa}).to_csv(
            AACRC_OUTPUT_DIR / f"aacrc_predictions_trial_{trial:03d}.csv", index=False
        )

    saturation = {
        METHOD_CRC: float(j == len(lam) - 1),
        METHOD_AACRC: deployment_diag["rate_at_lam_max"],
    }

    # --- Arm 3: ReCIRC -------------------------------------------------------
    if use_recirc:
        method_recirc = METHOD_RECIRC_TABICL if risk_model == "tabicl" else METHOD_RECIRC_HGB
        idx_rc, a_hat = run_recirc(
            F[idx_D], LOSS[idx_D], F[idx_C], L_C, F[idx_T], lam, alpha, seed, risk_model
        )
        outs[method_recirc] = (L_T[ar, idx_rc], W_T[ar, idx_rc])
        selected[method_recirc] = float(a_hat)
        saturation[method_recirc] = float(np.mean(idx_rc == len(lam) - 1))

    marginal_rows: List[Dict] = []
    conditional_rows: List[Dict] = []
    slice_rows: List[Dict] = []

    masks = slice_masks(data, idx_T)
    for method, (loss, width) in outs.items():
        r_sl = slice_risks(loss, data, idx_T)
        marginal_rows.append(
            {
                "method": method,
                "trial": trial,
                "seed": seed,
                "alpha": alpha,
                "selected_param": selected[method],
                "test_risk": float(loss.mean()),
                "excess_risk_event": float(loss.mean() > alpha),
                "worst_slice": float(r_sl.max()),
                "mean_excess_slice": float(np.maximum(r_sl - alpha, 0).mean()),
                "avg_width": float(width.mean()),
                "median_width": float(np.median(width)),
                "infinite_width_rate": float(np.isinf(width).mean()),
                "rate_at_lam_max": saturation[method],
                "n_test": int(len(idx_T)),
            }
        )

        for b in range(N_BINS):
            msk = bins_T == b
            conditional_rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "seed": seed,
                    "bin": int(b),
                    "n": int(msk.sum()),
                    "conditional_risk": float(loss[msk].mean()),
                    "mean_width": float(width[msk].mean()),
                }
            )

        for name, msk in masks.items():
            if msk.sum() >= SLICE_NMIN:
                slice_rows.append(
                    {
                        "method": method,
                        "trial": trial,
                        "seed": seed,
                        "slice": name,
                        "n": int(msk.sum()),
                        "slice_risk": float(loss[msk].mean()),
                        "mean_width": float(width[msk].mean()),
                    }
                )

    return marginal_rows, conditional_rows, slice_rows, crc_gap


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def width_mean(values):
    """Preserve infinite widths in aggregation across pandas versions."""
    return float(np.asarray(values, dtype=float).mean())


def width_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 and np.isfinite(values).all() else float("nan")


def aggregate_marginal(df_marginal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate marginal metrics by method across trials."""
    if len(df_marginal) == 0:
        return pd.DataFrame()
    return (
        df_marginal.groupby("method", as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            risk_mean=("test_risk", "mean"),
            risk_sd=("test_risk", "std"),
            excess_event_rate=("excess_risk_event", "mean"),
            worst_slice_mean=("worst_slice", "mean"),
            worst_slice_sd=("worst_slice", "std"),
            mean_excess_slice=("mean_excess_slice", "mean"),
            avg_width_mean=("avg_width", width_mean),
            infinite_width_rate=("infinite_width_rate", "mean"),
            rate_at_lam_max=("rate_at_lam_max", "mean"),
            avg_width_sd=("avg_width", width_sd),
        )
        .sort_values("worst_slice_mean")
    )


def aggregate_conditional(df_conditional: pd.DataFrame) -> pd.DataFrame:
    """Aggregate conditional risk and width by method and difficulty bin."""
    if len(df_conditional) == 0:
        return pd.DataFrame()
    return df_conditional.groupby(["method", "bin"], as_index=False).agg(
        mean_cond_risk=("conditional_risk", "mean"),
        sd_cond_risk=("conditional_risk", "std"),
        mean_width=("mean_width", width_mean),
        n_mean=("n", "mean"),
    )


def paired_comparison(df_marginal: pd.DataFrame, m1: str, m2: str, metric: str = "worst_slice") -> Optional[Dict]:
    """Paired difference m1 - m2 across trials (negative means m1 is better)."""
    if m1 not in df_marginal["method"].unique() or m2 not in df_marginal["method"].unique():
        return None
    a = df_marginal[df_marginal.method == m1].sort_values("trial")[metric].to_numpy()
    b = df_marginal[df_marginal.method == m2].sort_values("trial")[metric].to_numpy()
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


def build_paired_table(df_marginal: pd.DataFrame, methods_present: Sequence[str]) -> pd.DataFrame:
    """Assemble the paired comparisons that are meaningful given the arms that ran."""
    rows = []
    recirc = [m for m in (METHOD_RECIRC_TABICL, METHOD_RECIRC_HGB) if m in methods_present]
    for m in recirc:
        for metric in ("worst_slice", "test_risk", "avg_width"):
            r = paired_comparison(df_marginal, m, METHOD_AACRC, metric)
            if r:
                rows.append(r)
    for metric in ("worst_slice", "test_risk", "avg_width"):
        r = paired_comparison(df_marginal, METHOD_AACRC, METHOD_CRC, metric)
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

    out["marginal_risk"] = fmt("risk_mean", "risk_sd", 4)
    out["worst_slice_risk"] = fmt("worst_slice_mean", "worst_slice_sd", 4)
    out["avg_width"] = fmt("avg_width_mean", "avg_width_sd", 0)
    return out[
        [
            "method",
            "n_trials",
            "marginal_risk",
            "excess_event_rate",
            "worst_slice_risk",
            "mean_excess_slice",
            "avg_width",
            "infinite_width_rate",
        ]
    ]


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def save_plots(
    df_marginal: pd.DataFrame,
    df_conditional: pd.DataFrame,
    output_dir: str,
    n_obs: int,
    alpha: float = ALPHA,
) -> None:
    """Save the marginal four-panel figure and the conditional two-panel figure."""
    os.makedirs(output_dir, exist_ok=True)

    order = [METHOD_CRC, METHOD_AACRC, METHOD_RECIRC_TABICL, METHOD_RECIRC_HGB]
    methods_order = [m for m in order if m in df_marginal["method"].unique()]
    if not methods_order:
        return

    # --- Figure 1: marginal metrics, one dot per trial -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    panels = [
        ("test_risk", r"$\widehat{R}_{\mathrm{test}}$", "(a) Marginal risk", True),
        ("worst_slice", r"$\max_s \widehat{R}_s$", "(b) Worst slice (smoker×BMI / smoker×age)", True),
        ("mean_excess_slice", "Mean excess", "(c) Mean slice excess", False),
        ("avg_width", "Interval width (\\$)", "(d) Mean interval width", False),
    ]
    jit = np.random.default_rng(0)
    for ax, (metric, ylab, title, show_alpha) in zip(axes, panels):
        for j, m in enumerate(methods_order):
            tmp = df_marginal[df_marginal["method"] == m]
            x = np.full(len(tmp), j, float) + jit.normal(0, 0.04, len(tmp))
            ax.scatter(x, tmp[metric], alpha=0.75, s=45, color=PALETTE.get(m, "gray"))
            ax.hlines(tmp[metric].mean(), j - 0.2, j + 0.2, lw=2.5, color="black")
        if show_alpha:
            ax.axhline(alpha, ls="--", color="gray", lw=1.5)
        ax.set_xticks(range(len(methods_order)))
        ax.set_xticklabels(methods_order, fontsize=9, rotation=15)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35)
    fig.suptitle(
        f"Medical Insurance (n={n_obs}) | alpha={alpha} | {df_marginal['trial'].nunique()} splits",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "marginal_metrics_by_method.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: conditional risk and width across difficulty bins ---------
    agg = aggregate_conditional(df_conditional)
    if len(agg) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for m in methods_order:
        tmp = agg[agg["method"] == m].sort_values("bin")
        axes[0].errorbar(
            tmp["bin"],
            tmp["mean_cond_risk"],
            yerr=tmp["sd_cond_risk"].fillna(0),
            marker="o",
            capsize=3,
            label=m,
            color=PALETTE.get(m, "gray"),
        )
        axes[1].plot(tmp["bin"], tmp["mean_width"], marker="o", label=m, color=PALETTE.get(m, "gray"))
    axes[0].axhline(alpha, ls="--", color="gray", label=fr"$\alpha={alpha}$")
    axes[0].set_xlabel(r"Difficulty bin (scale $s_- + s_+$)")
    axes[0].set_ylabel("Conditional risk")
    axes[0].set_title("Conditional coverage")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Difficulty bin")
    axes[1].set_ylabel("Mean width (\\$)")
    axes[1].set_title("Interval width by bin")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "conditional_risk_and_width_by_bin.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimento 6: Medical Insurance — CRC marginal vs AA-CRC truncado em lambda=4 vs ReCIRC."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Diretório para salvar resultados.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Número de trials do experimento.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Seed base para randomização.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Nível de risco alvo.")
    parser.add_argument(
        "--risk-model", type=str, default=RISK_MODEL, choices=["tabicl", "hgb"], help="Modelo para ReCIRC."
    )
    parser.add_argument("--no-plots", action="store_true", help="Desabilita geração de gráficos.")
    parser.add_argument("--aacrc-repo", type=Path, default=AACRC_REPO)
    parser.add_argument("--aacrc-integration", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--aacrc-ridge", type=float, default=AACRC_RIDGE)
    parser.add_argument("--aacrc-maxiter", type=int, default=AACRC_MAXITER)
    parser.add_argument("--data-file", type=Path, default=Path(__file__).resolve().parent / "data_insurance/insurance.csv")
    parser.add_argument("--no-recirc", action="store_true", help="Executa apenas CRC e AA-CRC.")
    parser.add_argument("--self-check", action="store_true", help="Validação offline com NumPy/SciPy.")
    args = parser.parse_args()
    if args.trials < 1 or not 0 < args.alpha < 1:
        parser.error("--trials deve ser positivo e --alpha deve estar entre 0 e 1.")
    if not np.isfinite(args.aacrc_ridge) or args.aacrc_ridge < 0 or args.aacrc_maxiter < 1:
        parser.error("--aacrc-ridge deve ser finito e não negativo; --aacrc-maxiter deve ser positivo.")
    return args


def main():
    global AACRC_MODULE, AACRC_OUTPUT_DIR, AACRC_RIDGE, AACRC_MAXITER
    args = parse_args()

    alpha = args.alpha
    n_trials = args.trials
    base_seed = args.seed
    risk_model = args.risk_model

    output_dir = Path(args.output_dir) if args.output_dir else Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_recirc = not args.no_recirc
    if use_recirc and risk_model == "tabicl" and not HAS_TABICL:
        raise RuntimeError("TabICL indisponível. Instale tabicl, selecione --risk-model hgb ou --no-recirc.")
    AACRC_MODULE = load_original_aacrc(args.aacrc_repo, args.aacrc_integration)
    AACRC_OUTPUT_DIR = output_dir
    AACRC_RIDGE, AACRC_MAXITER = args.aacrc_ridge, args.aacrc_maxiter
    data = load_insurance(str(args.data_file))
    if int(FRAC_C * data.n) <= 1.0 / alpha:
        raise ValueError("Insurance precisa de len(C) > 1/alpha para ajustar AA-CRC.")
    print(f"insurance: {data.n} × {data.p} | n_D≈{int(FRAC_D * data.n)} (few-shot regime for the risk regressor)")
    print(f"device: {TABICL_DEVICE} | risk model: {risk_model if use_recirc else 'disabled'}")

    interaction = describe_interaction(data)
    print("\nsmoker×BMI interaction (charges):")
    print(interaction.round(0).to_string(index=False))
    interaction.to_csv(output_dir / "interaction_table.csv", index=False)

    rows_marg: List[Dict] = []
    rows_cond: List[Dict] = []
    rows_slice: List[Dict] = []
    gaps: List[float] = []

    t0 = time.time()
    for t in range(n_trials):
        seed = base_seed + t
        m_rows, c_rows, s_rows, gap = run_one_trial(
            trial=t, seed=seed, data=data, alpha=alpha, risk_model=risk_model, use_recirc=use_recirc
        )
        rows_marg.extend(m_rows)
        rows_cond.extend(c_rows)
        rows_slice.extend(s_rows)
        gaps.append(gap)

        pd.DataFrame(rows_marg).to_csv(output_dir / "marginal_results_incremental.csv", index=False)
        pd.DataFrame(rows_cond).to_csv(output_dir / "conditional_results_incremental.csv", index=False)

        if (t + 1) % 5 == 0 or (t + 1) == n_trials:
            print(f"trial {t + 1}/{n_trials} | {time.time() - t0:.0f}s acumulados")

    df_marginal = pd.DataFrame(rows_marg)
    df_conditional = pd.DataFrame(rows_cond)
    df_slices = pd.DataFrame(rows_slice)

    elapsed = time.time() - t0
    print(f"\ntotal: {elapsed:.0f}s | gap D→T médio (CRC marginal): {np.mean(gaps):+.3f}")

    df_marginal.to_csv(output_dir / "marginal_results.csv", index=False)
    df_conditional.to_csv(output_dir / "conditional_results.csv", index=False)
    df_slices.to_csv(output_dir / "slice_results.csv", index=False)

    summary = aggregate_marginal(df_marginal)
    cond_agg = aggregate_conditional(df_conditional)
    compact = build_compact_table(summary)
    paired = build_paired_table(df_marginal, df_marginal["method"].unique().tolist())

    summary.to_csv(output_dir / "summary_by_method.csv", index=False)
    cond_agg.to_csv(output_dir / "conditional_by_method_bin.csv", index=False)
    compact.to_csv(output_dir / "compact_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)

    print("\nResumo por método:")
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

    if not args.no_plots and np.isfinite(df_marginal["avg_width"]).all():
        save_plots(df_marginal, df_conditional, str(output_dir), n_obs=data.n, alpha=alpha)
        print(f"\nGráficos salvos em: {output_dir}")

    elif not args.no_plots:
        warnings.warn("Gráficos omitidos porque há larguras infinitas; consulte os CSVs.")

    meta = {
        "experiment": "experiment_6_insurance_official_aacrc2_truncated",
        "aacrc": {
            "source_commit": AACRC_COMMIT, "source_sha256": AACRC_SOURCE_SHA256,
            "repository": str(args.aacrc_repo.resolve()), "objective": "authors J/J_prime",
            "integration": args.aacrc_integration, "quadrature_nodes": 100,
            "optimizer": "SLSQP", "ridge": AACRC_RIDGE, "maxiter": AACRC_MAXITER,
            "fit_split": "C", "feature_standardization_split": "D",
            "sample_correction": "1/len(C)", "auxiliary_labels_per_observation": 5,
            "threshold_class": "affine u on standardized original one-hot covariates",
            "fit_interval_multiplier": "max(0, -log(u)) if u > 0 else infinity",
            "interval_multiplier": "min(LAM_MAX, max(0, -log(u))) if u > 0 else LAM_MAX",
            "deployment": {"truncate": True, "lam_max": float(LAM_MAX), "snap_to_grid": False},
            "theory_note": "original objective with postfit interval truncation; truncation can increase risk; no additional validity claim",
            "lambda_grid_applies": False,
        },
        "data_file": str(args.data_file.resolve()),
        "alpha": float(alpha),
        "n_trials": int(n_trials),
        "seed": int(base_seed),
        "risk_model": str(risk_model) if use_recirc else None,
        "recirc_enabled": bool(use_recirc),
        "tabicl_device": str(TABICL_DEVICE),
        "n_obs": int(data.n),
        "n_features": int(data.p),
        "frac_D": float(FRAC_D),
        "frac_C": float(FRAC_C),
        "loss_weights": {"w_neg": float(W_NEG), "w_pos": float(W_POS)},
        "quantiles": [float(Q_LO), float(Q_MED), float(Q_HI)],
        "lambda_grid": {"methods": ["CRC", "ReCIRC"], "n": int(N_LAM), "max": float(LAM_MAX), "n_train_anchors": int(N_LAM_TRAIN)},
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
        "n_bins": int(N_BINS),
        "mean_crc_train_test_gap": float(np.mean(gaps)) if gaps else None,
        "elapsed_seconds": float(elapsed),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
