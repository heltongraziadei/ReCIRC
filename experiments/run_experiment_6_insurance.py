#!/usr/bin/env python
"""Experiment 6: Medical Insurance — marginal CRC vs AA-CRC-linear (D u C) vs ReCIRC-TabICL.

This script reproduces the logic from the Experiment 6 notebook in a standalone
Python script. It uses the Medical Insurance dataset (1,338 x 6 mixed features,
target = annual medical charges) and compares three risk-control methods on top
of a quantile random forest (QRF) base predictor:

- Marginal CRC: a single global lambda calibrated on the calibration split C.
- AA-CRC-linear (D u C): a locally adaptive rule lambda(x) = Phi(x)^T beta fitted
  by least squares, whose offset is calibrated on the union D u C. This variant
  deliberately gives the linear baseline *more* data than the standard protocol.
- ReCIRC-TabICL: the risk surface R(lambda | x) is estimated directly with TabICL
  (context = the whole split D, ~535 rows) and the risk budget is calibrated on C.

The controlled loss is the asymmetric miscoverage loss
    L_lambda(x, y) = w_neg * 1{y < q_med - lambda * s_neg}
                   + w_pos * 1{y > q_med + lambda * s_pos},
with (w_neg, w_pos) = (0.2, 0.8) and s_neg, s_pos the lower/upper QRF scales.

Why this dataset. Medical Insurance exhibits the canonical non-linear interaction
smoker x BMI: the marginal effect of BMI on charges is roughly +$0.9k for
non-smokers and +$20k for smokers. A linear class for lambda(x) can capture the
marginal smoker effect (a dummy), but cannot let lambda depend on BMI differently
across smoking groups — a pure interaction, outside the AA-CRC-linear class.

Evaluation reports marginal risk, worst-slice risk over interaction slices
(smoker x obese and smoker x age), mean positive slice excess, average interval
width, and conditional risk across label-free difficulty bins (quintiles of the
total scale s_neg + s_pos).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Install required packages automatically
def ensure_packages():
    """Install required packages if they are not available.

    ``tabicl`` is treated as optional: if its installation fails the script still
    runs, either with the HGB backend or with the ReCIRC arm disabled.
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

# Asymmetric loss weights: under-coverage below the interval is penalised less
# than above it (heavy right tail of medical charges).
W_NEG, W_POS = 0.2, 0.8

# QRF quantile levels used to build the (lower, median, upper) scaffold.
Q_LO, Q_MED, Q_HI = 0.05, 0.50, 0.95

# Lambda grid used by every method (multiplier applied to the asymmetric scales).
N_LAM = 80
LAM_MAX = 4.0
EPS_SCALE = 1e-3

# Number of lambda anchors on which ReCIRC actually fits a risk regressor;
# the remaining grid points are recovered by monotone PCHIP interpolation.
N_LAM_TRAIN = 16

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
FALLBACK_TO_HGB_IF_TABICL_FAILS = False
TABICL_ESTIMATORS = 4
BATCH_ROWS = 4000

try:
    import torch

    TABICL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    TABICL_DEVICE = "cpu"

# Detect Google Drive mounted (Colab) and use as default
if os.path.isdir("/content/drive"):
    OUT_DIR = "/content/drive/MyDrive/PythonReCIRC/results/experiment_6_insurance_tabicl_aacrc_dcupc"
else:
    OUT_DIR = "insurance_crc_aacrc_dcupc_recirc_tabicl_results"

# Method labels (kept stable across CSV outputs and figures).
METHOD_CRC = "Marginal CRC"
METHOD_AACRC = "AA-CRC (D∪C)"
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
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
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
# AA-CRC-linear fitted on D u C
# -----------------------------------------------------------------------------


def aacrc_linear_fit(
    L_fit: np.ndarray,
    Phi_fit: np.ndarray,
    lam_grid: np.ndarray,
    alpha: float = ALPHA,
    B: float = 1.0,
) -> Tuple[LinearRegression, float]:
    """Fit lambda(x) = Phi(x)^T beta by least squares and calibrate a global offset.

    The regression target is the per-example loss-minimising lambda. The offset
    delta is the smallest shift on a symmetric grid for which the CRC bound
    evaluated on the same fitting sample falls below alpha. In this variant both
    the fit and the offset calibration use D u C.
    """
    lam_opt = lam_grid[np.argmin(L_fit, axis=1)]
    reg = LinearRegression().fit(Phi_fit, lam_opt)
    base = reg.predict(Phi_fit)
    n = L_fit.shape[0]

    def risk_at(d: float) -> float:
        lx = np.clip(base + d, lam_grid[0], lam_grid[-1])
        idx = np.searchsorted(lam_grid, lx).clip(0, len(lam_grid) - 1)
        return float(L_fit[np.arange(n), idx].mean())

    for d in np.linspace(-lam_grid[-1], lam_grid[-1], 400):
        if float(crc_upper_bound(np.asarray(risk_at(d)), n=n, B=B)) <= alpha:
            return reg, float(d)

    return reg, float(lam_grid[-1])


def aacrc_linear_predict(
    reg: LinearRegression, delta: float, Phi: np.ndarray, lam_grid: np.ndarray
) -> np.ndarray:
    """Map the fitted linear rule plus offset to lambda-grid indices."""
    lx = np.clip(reg.predict(Phi) + delta, lam_grid[0], lam_grid[-1])
    return np.searchsorted(lam_grid, lx).clip(0, len(lam_grid) - 1)


# -----------------------------------------------------------------------------
# ReCIRC
# -----------------------------------------------------------------------------


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
    a_max = float(L_C[:, 0].mean())
    a_grid = np.linspace(0, a_max, 201)
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

    # AA-CRC in this variant is fitted and calibrated on the union D u C.
    idx_AA = np.concatenate([idx_D, idx_C])
    L_AA = LOSS[idx_AA]
    Phi_AA = np.column_stack([np.ones(len(idx_AA)), data.X[idx_AA]])
    Phi_T = np.column_stack([np.ones(len(idx_T)), data.X[idx_T]])

    difficulty_T = (s_neg + s_pos)[idx_T]
    bins_T = equal_mass_bins(difficulty_T, N_BINS)

    outs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    selected: Dict[str, float] = {}

    # --- Arm 1: marginal CRC -------------------------------------------------
    j = crc_global(L_C, alpha)
    crc_gap = float(LOSS[idx_D][:, j].mean() - L_T[:, j].mean())
    outs[METHOD_CRC] = (L_T[:, j], W_T[:, j])
    selected[METHOD_CRC] = float(lam[j])

    # --- Arm 2: AA-CRC-linear on D u C ---------------------------------------
    reg, delta = aacrc_linear_fit(L_AA, Phi_AA, lam, alpha)
    idx_aa = aacrc_linear_predict(reg, delta, Phi_T, lam)
    outs[METHOD_AACRC] = (L_T[ar, idx_aa], W_T[ar, idx_aa])
    selected[METHOD_AACRC] = float(delta)

    # --- Arm 3: ReCIRC -------------------------------------------------------
    if use_recirc:
        method_recirc = METHOD_RECIRC_TABICL if risk_model == "tabicl" else METHOD_RECIRC_HGB
        idx_rc, a_hat = run_recirc(
            F[idx_D], LOSS[idx_D], F[idx_C], L_C, F[idx_T], lam, alpha, seed, risk_model
        )
        outs[method_recirc] = (L_T[ar, idx_rc], W_T[ar, idx_rc])
        selected[method_recirc] = float(a_hat)

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
                "alpha": alpha,
                "selected_param": selected[method],
                "test_risk": float(loss.mean()),
                "excess_risk_event": float(loss.mean() > alpha),
                "worst_slice": float(r_sl.max()),
                "mean_excess_slice": float(np.maximum(r_sl - alpha, 0).mean()),
                "avg_width": float(width.mean()),
                "median_width": float(np.median(width)),
                "n_test": int(len(idx_T)),
            }
        )

        for b in range(N_BINS):
            msk = bins_T == b
            conditional_rows.append(
                {
                    "method": method,
                    "trial": trial,
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
            avg_width_mean=("avg_width", "mean"),
            avg_width_sd=("avg_width", "std"),
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
        mean_width=("mean_width", "mean"),
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
        "sd_diff": float(d.std()),
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
            lambda r: f"{r[mean_col]:.{digits}f} ± {0 if pd.isna(r[sd_col]) else r[sd_col]:.{digits}f}",
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
            yerr=tmp["sd_cond_risk"],
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
        description="Experiment 6: Medical Insurance — marginal CRC vs AA-CRC-linear (D∪C) vs ReCIRC."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory in which to save results.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Number of experiment trials.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base seed for randomization.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Target risk level.")
    parser.add_argument(
        "--risk-model", type=str, default=RISK_MODEL, choices=["tabicl", "hgb"], help="Model for ReCIRC."
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    alpha = args.alpha
    n_trials = args.trials
    base_seed = args.seed
    risk_model = args.risk_model

    output_dir = Path(args.output_dir) if args.output_dir else Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether the ReCIRC arm can run at all.
    use_recirc = True
    if risk_model == "tabicl" and not HAS_TABICL:
        if FALLBACK_TO_HGB_IF_TABICL_FAILS:
            warnings.warn("TabICL unavailable; using HGB for the ReCIRC arm.")
            risk_model = "hgb"
        else:
            warnings.warn("TabICL unavailable; the ReCIRC arm will be omitted.")
            use_recirc = False

    data = load_insurance()
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
    print(f"\ntotal: {elapsed:.0f}s | mean D→T gap (marginal CRC): {np.mean(gaps):+.3f}")

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

    print("\nSummary by method:")
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
        save_plots(df_marginal, df_conditional, str(output_dir), n_obs=data.n, alpha=alpha)
        print(f"\nPlots saved to: {output_dir}")

    meta = {
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
        "lambda_grid": {"n": int(N_LAM), "max": float(LAM_MAX), "n_train_anchors": int(N_LAM_TRAIN)},
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

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
