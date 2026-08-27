#!/usr/bin/env python
"""Superconductor mechanism: ReCIRC-TabICL vs AA-CRC-linear (D u C) vs marginal CRC.

This script reproduces the logic from the Superconductor mechanism notebook in a
standalone Python script. It is the realistic counterpart of the XOR probe: the
UCI Superconductivity dataset has ~21,263 materials and 81 physical features,
with the critical temperature as target. Predictability may vary non-linearly
across families of materials, but no simple interaction is declared as in the
XOR generator — so this is a less controlled, more realistic test of the same
thesis: ReCIRC estimates R(x, lambda) with a flexible regressor while
AA-CRC-linear stays restricted to the class lambda(x) = Phi(x)^T beta.

Evaluation slices are the 4x4 quantile-bin interactions of the two features most
correlated with the target; they are never declared to any method.

Arms compared (identical splits, paired comparisons):

- CRC: a single global lambda calibrated on the calibration split C.
- AA-CRC-lin (D u C): lambda(x) = Phi(x)^T beta fitted by least squares on the
  union D u C, with a global shift also calibrated on D u C. This variant
  deliberately gives the linear baseline more data than the standard protocol.
- ReCIRC: the risk surface R(x, lambda) is estimated directly with TabICL
  (context = D, subsampled to N_D_TABICL rows) and the risk budget is calibrated
  on C.

The controlled loss is the asymmetric miscoverage loss
    L_lambda(x, y) = w_neg * 1{y < q_med - lambda * s_neg}
                   + w_pos * 1{y > q_med + lambda * s_pos},
with (w_neg, w_pos) = (0.2, 0.8) and s_neg, s_pos the lower/upper QRF scales.

Reported metrics: marginal risk, average interval width, worst-slice risk,
slice CVaR (mean of the top decile of slices), mean positive slice excess, and
the fraction of slices exceeding alpha.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# Install required packages automatically
def ensure_packages():
    """Install required packages if they are not available.

    ``tabicl`` is treated as optional: if its installation fails the script still
    runs with the HistGB backend.
    """
    packages = {
        "numpy": ("numpy", True),
        "pandas": ("pandas", True),
        "scipy": ("scipy", True),
        "sklearn": ("scikit-learn", True),
        "matplotlib": ("matplotlib", True),
        "quantile_forest": ("quantile-forest", True),
        "tabicl": ("tabicl", False),
    }

    for import_name, (pip_name, required) in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"Instalando {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            except subprocess.CalledProcessError:
                if required:
                    raise
                print(f"warning: could not install {pip_name} (optional package).")


ensure_packages()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

try:
    from tabicl import TabICLRegressor

    HAS_TABICL = True
except Exception as _tabicl_import_error:  # pragma: no cover - environment dependent
    HAS_TABICL = False
    print("warning: tabicl unavailable —", _tabicl_import_error)


# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------

ALPHA = 0.10
N_TRIALS = 20
BASE_SEED = 42

# Asymmetric loss weights.
W_NEG, W_POS = 0.2, 0.8

# QRF quantile levels used to build the (lower, median, upper) scaffold.
Q_LO, Q_MED, Q_HI = 0.05, 0.50, 0.95

# Lambda grid used by every method (multiplier applied to the asymmetric scales).
N_LAM = 80
LAM_MAX = 4.0
EPS_SCALE = 1e-3

# Number of lambda anchors on which ReCIRC actually fits a risk regressor;
# the remaining grid points come from monotone PCHIP interpolation.
N_LAM_TRAIN = 16

# Split fractions: D = risk/context, C = calibration, T = test (the remainder).
FRAC_D, FRAC_C = 0.40, 0.30

# QRF hyperparameters.
N_TREES_QRF = 200
MAX_DEPTH_QRF = 12
MIN_LEAF_QRF = 40

# Minimum slice size for a slice to enter the slice statistics.
SLICE_NMIN = 50

# Number of quantile bins per slicing feature (slices are the 4x4 interactions).
N_SLICE_BINS = 4

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
        "/content/drive/MyDrive/PythonReCIRC/results/experiment_mechanism_superconductor_tabicl_aacrc_dcupc"
    )
else:
    OUT_DIR = "mechanism_superconductor_crc_aacrc_dcupc_recirc_tabicl_results"

# Method labels (kept stable across CSV outputs and figures).
METHOD_CRC = "CRC"
METHOD_AACRC = "AA-CRC-lin (D∪C)"
METHOD_RECIRC = "ReCIRC"
METHODS_ORDER = [METHOD_CRC, METHOD_AACRC, METHOD_RECIRC]

PALETTE = {
    METHOD_CRC: "#888888",
    METHOD_AACRC: "#d62728",
    METHOD_RECIRC: "#1f77b4",
}


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
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if not os.path.exists(cache_file):
        wd = tempfile.mkdtemp(prefix="super_")
        zp = os.path.join(wd, "s.zip")
        ok = False
        for url in (SUPER_URL, SUPER_FALLBACK):
            try:
                print("baixando", url, "...")
                urllib.request.urlretrieve(url, zp)
                ok = True
                break
            except Exception as e:
                print("  failed:", str(e)[:70])
        if not ok:
            raise RuntimeError("could not download the Superconductor dataset")

        with zipfile.ZipFile(zp) as zf:
            zf.extractall(wd)

        train_path = None
        for r, _, fs in os.walk(wd):
            for f in fs:
                if f == "train.csv":
                    train_path = os.path.join(r, f)
        if train_path is None:
            raise RuntimeError("train.csv not found in the downloaded archive")

        pd.read_csv(train_path).to_csv(cache_file, index=False)

    df = pd.read_csv(cache_file)
    y = df["critical_temp"].to_numpy(np.float32)
    X = df.drop(columns=["critical_temp"]).to_numpy(np.float32)
    names = list(df.drop(columns=["critical_temp"]).columns)
    print(
        f"Superconductor: {X.shape[0]} × {X.shape[1]} | median temp_crit={np.median(y):.1f} max={y.max():.1f}"
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
    are the resulting bin interactions. These slices are label-free and are never
    exposed to any of the methods.
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the QRF scaffold on D and predict the (lo, med, hi) quantiles for all rows.

    The lower and upper quantiles are clipped against the median so that the
    resulting asymmetric scales are non-negative by construction.
    """
    qrf = RandomForestQuantileRegressor(
        n_estimators=N_TREES_QRF,
        max_depth=MAX_DEPTH_QRF,
        min_samples_leaf=MIN_LEAF_QRF,
        n_jobs=-1,
        random_state=seed,
    ).fit(X_all[idx_D], y_all[idx_D])

    Q = qrf.predict(X_all, quantiles=[Q_LO, Q_MED, Q_HI])
    lo = np.minimum(Q[:, 0], Q[:, 1]).astype(np.float32)
    hi = np.maximum(Q[:, 2], Q[:, 1]).astype(np.float32)
    med = Q[:, 1].astype(np.float32)
    return lo, med, hi


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
# AA-CRC-linear fitted on D u C
# -----------------------------------------------------------------------------


def aacrc_linear_fit(
    L_fit: np.ndarray,
    Phi_fit: np.ndarray,
    lam_grid: np.ndarray,
    alpha: float = ALPHA,
    B: float = 1.0,
) -> Tuple[LinearRegression, float]:
    """Fit lambda(x) = Phi(x)^T beta by least squares and calibrate a global shift.

    The regression target is the smallest lambda that zeroes the loss of each
    point on the fitting split. The shift delta is the smallest value on a
    symmetric grid for which the CRC bound falls below alpha, which yields the
    narrowest admissible intervals. In this variant both the fit and the shift
    calibration use D u C.
    """
    lam_opt = lam_grid[np.argmin(L_fit, axis=1)]
    reg = LinearRegression().fit(Phi_fit, lam_opt)  # LINEAR class in the features
    base = reg.predict(Phi_fit)
    n = L_fit.shape[0]

    def risk_at(delta: float) -> float:
        lam_x = np.clip(base + delta, lam_grid[0], lam_grid[-1])
        idx = np.searchsorted(lam_grid, lam_x).clip(0, len(lam_grid) - 1)
        return float(L_fit[np.arange(n), idx].mean())

    for d in np.linspace(-lam_grid[-1], lam_grid[-1], 400):
        if float(crc_upper_bound(np.asarray(risk_at(d)), n=n, B=B)) <= alpha:
            return reg, float(d)

    return reg, float(lam_grid[-1])


def aacrc_linear_predict(
    reg: LinearRegression, delta: float, Phi: np.ndarray, lam_grid: np.ndarray
) -> np.ndarray:
    """Map the fitted linear rule plus shift to lambda-grid indices."""
    lam_x = np.clip(reg.predict(Phi) + delta, lam_grid[0], lam_grid[-1])
    return np.searchsorted(lam_grid, lam_x).clip(0, len(lam_grid) - 1)


# -----------------------------------------------------------------------------
# ReCIRC (Route 2: free multivariate risk regressor)
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
) -> Tuple[np.ndarray, float]:
    """Fit the ReCIRC risk surface on D and calibrate the risk budget on C.

    One risk model is fitted per lambda anchor. For the TabICL backend the
    context D is subsampled to N_D_TABICL rows, since in-context performance
    saturates early while attention cost is quadratic in the context length.

    Returns the selected lambda index per test point and the calibrated budget.
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
    a_max = float(L_C[:, 0].mean())
    a_grid = np.linspace(0, a_max, 201)
    risks = np.array([L_C[ar, invert_risk(R_C, a)].mean() for a in a_grid])

    nC = F_C.shape[0]
    bound = crc_upper_bound(risks, n=nC, B=1.0)
    v = np.where(bound <= alpha)[0]
    a_hat = a_grid[v[-1]] if len(v) else a_grid[0]

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
) -> Tuple[List[Dict], List[Dict]]:
    """Run the three arms on a single split and collect summary and slice rows."""
    n = len(X)
    idx_D, idx_C, idx_T = make_three_way_split(n, seed)

    q_lo, q_med, q_hi = fit_qrf(X, Y, idx_D, seed)
    LOSS, WIDTH, lam, s_neg, s_pos = build_loss(Y, q_lo, q_med, q_hi)
    F = feats(X, q_med, s_neg, s_pos)

    L_C, L_T, W_T = LOSS[idx_C], LOSS[idx_T], WIDTH[idx_T]
    ar = np.arange(len(idx_T))

    # AA-CRC in this variant is fitted and calibrated on the union D u C.
    idx_AA = np.concatenate([idx_D, idx_C])
    L_AA = LOSS[idx_AA]
    Phi_AA = np.column_stack([np.ones(len(idx_AA)), X[idx_AA]])
    Phi_T = np.column_stack([np.ones(len(idx_T)), X[idx_T]])

    sl = slice_fn(idx_T)

    outs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    selected: Dict[str, float] = {}

    # --- Arm 1: marginal CRC, calibrated on C only ---------------------------
    j = crc_global(L_C, alpha)
    outs[METHOD_CRC] = (L_T[:, j], W_T[:, j])
    selected[METHOD_CRC] = float(lam[j])

    # --- Arm 2: AA-CRC-linear, fitted and calibrated on D u C ----------------
    reg, delta = aacrc_linear_fit(L_AA, Phi_AA, lam, alpha)
    idx_aa = aacrc_linear_predict(reg, delta, Phi_T, lam)
    outs[METHOD_AACRC] = (L_T[ar, idx_aa], W_T[ar, idx_aa])
    selected[METHOD_AACRC] = float(delta)

    # --- Arm 3: ReCIRC, risk model on D, budget calibrated on C --------------
    idx_rc, a_hat = run_recirc(F[idx_D], LOSS[idx_D], F[idx_C], L_C, F[idx_T], lam, alpha, seed, backend)
    outs[METHOD_RECIRC] = (L_T[ar, idx_rc], W_T[ar, idx_rc])
    selected[METHOD_RECIRC] = float(a_hat)

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
                "risk_backend": backend,
                "selected_param": selected[method],
                "marginal_risk": float(loss.mean()),
                "excess_risk_event": float(loss.mean() > alpha),
                "avg_width": float(width.mean()),
                "median_width": float(np.median(width)),
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
            avg_width_mean=("avg_width", "mean"),
            avg_width_sd=("avg_width", "std"),
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
        mean_width=("mean_width", "mean"),
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
        "sd_diff": float(d.std()),
        "win_rate_first": float((d < 0).mean()),
    }


def build_paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the paired comparisons reported in the notebook, plus width."""
    rows = []
    for metric in ("worst_slice", "slice_cvar", "marginal_risk", "avg_width"):
        for m1, m2 in ((METHOD_RECIRC, METHOD_AACRC), (METHOD_AACRC, METHOD_CRC)):
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
            lambda r: f"{r[mean_col]:.{digits}f} ± {0 if pd.isna(r[sd_col]) else r[sd_col]:.{digits}f}",
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
        stds = [df[df.method == m][col].std() for m in methods]
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
        description="Superconductor mechanism experiment: CRC vs AA-CRC-linear (D∪C) vs ReCIRC."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory in which to save results.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Number of experiment trials.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base seed for split randomization.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Target risk level.")
    parser.add_argument(
        "--slice-bins", type=int, default=N_SLICE_BINS, help="Bins per feature in the slice definition."
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default=RISK_BACKEND,
        choices=["tabicl", "histgb"],
        help="Backbone for the ReCIRC risk regressor.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    alpha = args.alpha
    n_trials = args.trials
    base_seed = args.seed
    backend = args.risk_model

    output_dir = Path(args.output_dir) if args.output_dir else Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "tabicl" and not HAS_TABICL:
        print("tabicl unavailable -> histgb")
        backend = "histgb"
    if backend == "tabicl" and TABICL_DEVICE == "cpu":
        warnings.warn("TabICL on CPU (without GPU) — slow. Colab: Runtime → Change runtime type → GPU.")

    print(
        f"config OK | risk regressor: {backend} | device: {TABICL_DEVICE if backend == 'tabicl' else 'n/a'}"
    )

    X, Y, names = load_superconductor()
    slice_fn, slice_features = build_slice_function(X, Y, names, n_bins=args.slice_bins)

    summary_rows: List[Dict] = []
    slice_rows: List[Dict] = []

    t0 = time.time()
    for t in range(n_trials):
        seed = base_seed + t
        s_rows, sl_rows = run_one_trial(
            trial=t, seed=seed, X=X, Y=Y, slice_fn=slice_fn, alpha=alpha, backend=backend
        )
        summary_rows.extend(s_rows)
        slice_rows.extend(sl_rows)

        pd.DataFrame(summary_rows).to_csv(output_dir / "trial_results_incremental.csv", index=False)

        if (t + 1) % 5 == 0 or (t + 1) == n_trials:
            print(f"trial {t + 1}/{n_trials} | {time.time() - t0:.0f}s acumulados")

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

    print(f"\n=== Superconductor — AA-CRC-linear (D∪C) — {n_trials} trials | {elapsed:.0f}s ===\n")
    print(summary.round(4).to_string(index=False))
    print("\nTabela compacta:")
    print(compact.to_string(index=False))

    if len(paired):
        print("\nPaired comparisons (negative = first method is better):")
        for _, r in paired.iterrows():
            print(
                f"  [{r['metric']}] {r['comparison']}: {r['mean_diff']:+.4f}±{r['sd_diff']:.4f} | "
                f"first is better in {r['win_rate_first'] * 100:.0f}% of trials"
            )

    if not args.no_plots:
        save_plots(df, df_slices, str(output_dir), slice_features=slice_features, alpha=alpha)
        print(f"\nPlots saved to: {output_dir}")

    meta = {
        "experiment": "mechanism_superconductor",
        "alpha": float(alpha),
        "n_trials": int(n_trials),
        "seed": int(base_seed),
        "n_obs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "slice_features": list(slice_features),
        "slice_bins": int(args.slice_bins),
        "risk_backend": str(backend),
        "tabicl_device": str(TABICL_DEVICE),
        "tabicl_context_size": int(N_D_TABICL),
        "frac_D": float(FRAC_D),
        "frac_C": float(FRAC_C),
        "loss_weights": {"w_neg": float(W_NEG), "w_pos": float(W_POS)},
        "quantiles": [float(Q_LO), float(Q_MED), float(Q_HI)],
        "lambda_grid": {"n": int(N_LAM), "max": float(LAM_MAX), "n_train_anchors": int(N_LAM_TRAIN)},
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

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
