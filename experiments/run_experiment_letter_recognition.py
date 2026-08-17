#!/usr/bin/env python
"""Experiment 5: Letter Recognition — CRC vs AA-CRC-style vs ReCIRC-TabICL.

This script reproduces the logic from the Experiment 5 notebook in a
standalone Python script. It uses the UCI Letter Recognition dataset,
trains a simple multinomial base classifier and compares three methods:

- Standard CRC
- AA-CRC-style
- ReCIRC (estimating R(m | x) directly with TabICL or HGB)

The goal is to control the top-m classification loss:
    L_m(x, y) = 1{rank_y(x) > m}

Evaluation includes marginal risk and per-difficulty-bin analyses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Install required packages automatically
def ensure_packages():
    """Install required packages if they are not available."""
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
        "tabicl": "tabicl",
    }
    
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"Instalando {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_packages()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------

ALPHA = 0.10
N_TRIALS = 10
BASE_SEED = 42

BASE_TRAIN_FRAC = 0.30
RISK_FRAC = 0.30
CAL_FRAC = 0.20
TEST_FRAC = 0.20

RISK_MODEL = "tabicl"  # "tabicl" or "hgb"
FALLBACK_TO_HGB_IF_TABICL_FAILS = False
K_AUG_PER_EXAMPLE = 8
MAX_RISK_TRAIN_ROWS = 6000
A_GRID = np.linspace(0.0, 1.0, 81)
T_GRID = np.linspace(-6.0, 6.0, 121)
N_BINS = 5

# Detect Google Drive mounted (Colab) and use as default
if os.path.isdir("/content/drive"):
    OUT_DIR = "/content/drive/MyDrive/PythonReCIRC/results/experiment_5_letter_recognition"
else:
    OUT_DIR = "letter_recognition_crc_aacrc_recirc_tabicl_results"

BASE_CLF_MAX_ITER = 1500
BASE_CLF_C = 2.0


# -----------------------------------------------------------------------------
# Mathematical helpers
# -----------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def true_label_ranks(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Rank 1 => true class most likely; Rank K => least likely."""
    order = np.argsort(-proba, axis=1)
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(1, order.shape[1] + 1)
    return inv[np.arange(len(y_true)), y_true].astype(int)


def losses_from_m_values(ranks: np.ndarray, m_values: Iterable[int]) -> np.ndarray:
    m_values = np.asarray(m_values, dtype=int)
    return (ranks[:, None] > m_values[None, :]).astype(np.float32)


def losses_from_m_matrix(ranks: np.ndarray, m_matrix: np.ndarray) -> np.ndarray:
    return (ranks[:, None] > m_matrix).astype(np.float32)


def score_features(X_raw: np.ndarray, proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float32)
    sortp = np.sort(proba, axis=1)[:, ::-1]
    top5 = sortp[:, :5]
    entropy = -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1) / np.log(proba.shape[1])
    margin12 = sortp[:, 0] - sortp[:, 1]
    mass_top3 = sortp[:, :3].sum(axis=1)
    mass_top5 = sortp[:, :5].sum(axis=1)
    n_above_005 = (proba >= 0.05).sum(axis=1) / proba.shape[1]
    n_above_010 = (proba >= 0.10).sum(axis=1) / proba.shape[1]
    prob_std = proba.std(axis=1)

    return np.column_stack([
        X_raw.astype(np.float32),
        top5,
        entropy,
        margin12,
        mass_top3,
        mass_top5,
        n_above_005,
        n_above_010,
        prob_std,
    ]).astype(np.float32)


def difficulty_score_from_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float32)
    return -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1) / np.log(proba.shape[1])


def make_equal_mass_bins(score: np.ndarray, n_bins: int = 5) -> np.ndarray:
    score = np.asarray(score)
    ranks = pd.Series(score).rank(method="first").values
    bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(bins, dtype=int)


# -----------------------------------------------------------------------------
# CRC calibration
# -----------------------------------------------------------------------------


def crc_upper_bound(mean_loss: np.ndarray, n: int, B: float = 1.0) -> np.ndarray:
    return (n / (n + 1.0)) * mean_loss + B / (n + 1.0)


def select_smallest_protective_param(params: np.ndarray, losses_cal: np.ndarray, alpha: float = ALPHA, B: float = 1.0):
    params = np.asarray(params)
    n = losses_cal.shape[0]
    mean_losses = losses_cal.mean(axis=0)
    bounds = crc_upper_bound(mean_losses, n=n, B=B)
    ok = bounds <= alpha
    if ok.any():
        j = int(np.where(ok)[0][0])
    else:
        j = len(params) - 1
    path = pd.DataFrame({
        "param": params,
        "mean_cal_loss": mean_losses,
        "crc_bound": bounds,
        "passes": ok,
    })
    return params[j], j, path


def select_largest_budget_param(params: np.ndarray, losses_cal: np.ndarray, alpha: float = ALPHA, B: float = 1.0):
    params = np.asarray(params)
    n = losses_cal.shape[0]
    mean_losses = losses_cal.mean(axis=0)
    bounds = crc_upper_bound(mean_losses, n=n, B=B)
    ok = bounds <= alpha
    if ok.any():
        j = int(np.where(ok)[0][-1])
    else:
        j = 0
    path = pd.DataFrame({
        "param": params,
        "mean_cal_loss": mean_losses,
        "crc_bound": bounds,
        "passes": ok,
    })
    return params[j], j, path


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

LETTER_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
FEATURE_NAMES = [
    "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
    "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]


def load_letter_recognition(cache_file: str = "./data_letter_recognition/letter-recognition.csv"):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if not os.path.exists(cache_file):
        print("Downloading Letter Recognition from UCI...")
        df = pd.read_csv(LETTER_URL, header=None)
        df.to_csv(cache_file, index=False)
    else:
        df = pd.read_csv(cache_file)

    if df.shape[1] != 17:
        raise ValueError(f"Esperava 17 colunas; vieram {df.shape[1]}")

    df.columns = ["letter"] + FEATURE_NAMES
    le = LabelEncoder()
    y = le.fit_transform(df["letter"].astype(str).values)
    X = df[FEATURE_NAMES].astype(np.float32).values
    class_names = list(le.classes_)
    return X, y, class_names, df


# -----------------------------------------------------------------------------
# Splits and base classifier
# -----------------------------------------------------------------------------


def make_four_way_split(y: np.ndarray, seed: int, base_train_frac: float = BASE_TRAIN_FRAC, risk_frac: float = RISK_FRAC,
                       cal_frac: float = CAL_FRAC, test_frac: float = TEST_FRAC):
    idx = np.arange(len(y))

    idx_base, idx_rest = train_test_split(
        idx,
        train_size=base_train_frac,
        random_state=seed,
        stratify=y,
    )

    rest_frac = 1.0 - base_train_frac
    risk_rel = risk_frac / rest_frac
    idx_risk, idx_rest2 = train_test_split(
        idx_rest,
        train_size=risk_rel,
        random_state=seed + 17,
        stratify=y[idx_rest],
    )

    cal_rel = cal_frac / (cal_frac + test_frac)
    idx_cal, idx_test = train_test_split(
        idx_rest2,
        train_size=cal_rel,
        random_state=seed + 31,
        stratify=y[idx_rest2],
    )

    return idx_base, idx_risk, idx_cal, idx_test


def fit_base_classifier(X: np.ndarray, y: np.ndarray, idx_base: np.ndarray, seed: int, K: int):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=BASE_CLF_C,
            max_iter=BASE_CLF_MAX_ITER,
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1,
            random_state=seed,
        ),
    )
    clf.fit(X[idx_base], y[idx_base])

    raw = clf.predict_proba(X)
    classes = clf.named_steps["logisticregression"].classes_.astype(int)
    proba = np.zeros((len(X), K), dtype=np.float32)
    proba[:, classes] = raw.astype(np.float32)
    proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    return clf, proba


# -----------------------------------------------------------------------------
# AA-CRC-style
# -----------------------------------------------------------------------------


def fit_aacrc_style(F_risk: np.ndarray, ranks_risk: np.ndarray, K: int):
    target = (ranks_risk - 0.5) / K
    target = logit(target)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-4, 3, 30)),
    )
    model.fit(F_risk, target)
    return model


def aacrc_m_from_offset(psi: np.ndarray, t_grid: np.ndarray, K: int):
    psi = np.asarray(psi, dtype=np.float32)
    t_grid = np.asarray(t_grid, dtype=np.float32)
    rho = sigmoid(psi[:, None] + t_grid[None, :])
    m = np.ceil(K * rho).astype(int)
    return np.clip(m, 1, K)


def run_aacrc_style(F_risk: np.ndarray, ranks_risk: np.ndarray, F_cal: np.ndarray, ranks_cal: np.ndarray,
                    F_test: np.ndarray, ranks_test: np.ndarray, trial: int, K: int, alpha: float = ALPHA):
    model = fit_aacrc_style(F_risk, ranks_risk, K)

    psi_cal = model.predict(F_cal)
    m_cal_grid = aacrc_m_from_offset(psi_cal, T_GRID, K)
    losses_cal = losses_from_m_matrix(ranks_cal, m_cal_grid)

    t_hat, j_hat, path = select_smallest_protective_param(T_GRID, losses_cal, alpha=alpha)
    path["trial"] = trial
    path["method"] = "aa_crc_style"

    psi_test = model.predict(F_test)
    m_test = aacrc_m_from_offset(psi_test, np.asarray([t_hat]), K).ravel()
    losses_test = (ranks_test > m_test).astype(np.float32)

    return {
        "method": "aa_crc_style",
        "selected_param": float(t_hat),
        "selected_m_mean": float(np.mean(m_test)),
        "test_risk": float(np.mean(losses_test)),
        "avg_set_size": float(np.mean(m_test)),
        "median_set_size": float(np.median(m_test)),
        "losses_test": losses_test,
        "sizes_test": m_test.astype(float),
        "calibration_path": path,
    }


# -----------------------------------------------------------------------------
# ReCIRC
# -----------------------------------------------------------------------------


def make_tabicl_regressor():
    try:
        from tabicl import TabICLRegressor
    except Exception:
        try:
            from tabicl.sklearn import TabICLRegressor
        except Exception as e:
            raise ImportError("Não consegui importar TabICLRegressor. Tente: pip install tabicl") from e

    try:
        return TabICLRegressor()
    except TypeError:
        return TabICLRegressor


def build_augmented_risk_data(F: np.ndarray, ranks: np.ndarray, rng: np.random.Generator,
                             k_aug: int = K_AUG_PER_EXAMPLE, max_rows: int = MAX_RISK_TRAIN_ROWS, K_grid: np.ndarray = None):
    if K_grid is None:
        K_grid = np.arange(1, F.shape[0] if False else 0, dtype=int)
    # For this experiment, the m space is 1..K.
    # Keep the grid visible at runtime using the global K.
    global K
    m_grid = np.arange(1, K + 1, dtype=int)
    n = len(ranks)
    sampled_m = rng.choice(m_grid, size=(n, k_aug), replace=True)

    F_rep = np.repeat(F, k_aug, axis=0)
    m_flat = sampled_m.ravel().astype(int)
    rho_flat = (m_flat / K).astype(np.float32)
    y_aug = (np.repeat(ranks, k_aug) > m_flat).astype(np.float32)
    X_aug = np.column_stack([F_rep, rho_flat]).astype(np.float32)

    if max_rows is not None and len(X_aug) > max_rows:
        take = rng.choice(len(X_aug), size=max_rows, replace=False)
        X_aug = X_aug[take]
        y_aug = y_aug[take]

    return X_aug, y_aug


def fit_risk_estimator(F_risk: np.ndarray, ranks_risk: np.ndarray, seed: int, K_value: int = None, risk_model: str = RISK_MODEL):
    if K_value is None:
        K_value = F_risk.shape[0] if False else 0
    global K
    if K_value:
        K_local = K_value
    else:
        K_local = K

    rng = np.random.default_rng(seed)
    X_aug, y_aug = build_augmented_risk_data(F_risk, ranks_risk, rng, k_aug=K_AUG_PER_EXAMPLE, max_rows=MAX_RISK_TRAIN_ROWS, K_grid=np.arange(1, K_local + 1))

    scaler = StandardScaler()
    X_aug_s = scaler.fit_transform(X_aug)

    if risk_model == "tabicl":
        try:
            model = make_tabicl_regressor()
            model.fit(X_aug_s, y_aug)
            model_name = "tabicl"
        except Exception as e:
            if not FALLBACK_TO_HGB_IF_TABICL_FAILS:
                raise
            warnings.warn(f"TabICL falhou; usando HGB. Erro: {repr(e)}")
            model = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            )
            model.fit(X_aug_s, y_aug)
            model_name = "hgb_fallback"
    elif risk_model == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        )
        model.fit(X_aug_s, y_aug)
        model_name = "hgb"
    else:
        raise ValueError("risk_model must be 'tabicl' or 'hgb'.")

    return {"model": model, "scaler": scaler, "model_name": model_name}


def predict_risk_table(estimator: Dict, F_query: np.ndarray, batch_size: int = 5000):
    n = len(F_query)
    global K
    m_grid = np.arange(1, K + 1, dtype=int)
    F_rep = np.repeat(F_query, K, axis=0)
    rho = np.tile(m_grid / K, n).astype(np.float32)
    Xq = np.column_stack([F_rep, rho]).astype(np.float32)
    Xq_s = estimator["scaler"].transform(Xq)

    preds = []
    model = estimator["model"]
    for start in range(0, len(Xq_s), batch_size):
        end = start + batch_size
        preds.append(np.asarray(model.predict(Xq_s[start:end]), dtype=np.float32))
    risk = np.concatenate(preds).reshape(n, K)
    risk = np.clip(risk, 0.0, 1.0)
    risk[:, -1] = 0.0
    risk = np.minimum.accumulate(risk, axis=1)
    return risk


def m_from_risk_budget(risk_table: np.ndarray, a_grid: np.ndarray):
    n, k = risk_table.shape
    out = np.empty((n, len(a_grid)), dtype=int)
    for j, a in enumerate(a_grid):
        ok = risk_table <= a
        first = np.argmax(ok, axis=1)
        no_ok = ~ok.any(axis=1)
        first[no_ok] = k - 1
        out[:, j] = np.arange(1, k + 1)[first]
    return out


def run_recirc(F_risk: np.ndarray, ranks_risk: np.ndarray, F_cal: np.ndarray, ranks_cal: np.ndarray,
               F_test: np.ndarray, ranks_test: np.ndarray, trial: int, seed: int, alpha: float = ALPHA,
               risk_model: str = RISK_MODEL):
    estimator = fit_risk_estimator(F_risk, ranks_risk, seed=seed, risk_model=risk_model)

    risk_cal = predict_risk_table(estimator, F_cal)
    m_cal_by_a = m_from_risk_budget(risk_cal, A_GRID)
    losses_cal = losses_from_m_matrix(ranks_cal, m_cal_by_a)

    a_hat, j_hat, path = select_largest_budget_param(A_GRID, losses_cal, alpha=alpha)
    path["trial"] = trial
    path["method"] = f"recirc_{estimator['model_name']}"

    risk_test = predict_risk_table(estimator, F_test)
    m_test = m_from_risk_budget(risk_test, np.asarray([a_hat])).ravel()
    losses_test = (ranks_test > m_test).astype(np.float32)

    return {
        "method": f"recirc_{estimator['model_name']}",
        "selected_param": float(a_hat),
        "selected_m_mean": float(np.mean(m_test)),
        "test_risk": float(np.mean(losses_test)),
        "avg_set_size": float(np.mean(m_test)),
        "median_set_size": float(np.median(m_test)),
        "losses_test": losses_test,
        "sizes_test": m_test.astype(float),
        "calibration_path": path,
    }


# -----------------------------------------------------------------------------
# Standard CRC and evaluation by bins
# -----------------------------------------------------------------------------


def run_standard_crc(ranks_cal: np.ndarray, ranks_test: np.ndarray, trial: int, K: int):
    losses_cal = losses_from_m_values(ranks_cal, np.arange(1, K + 1, dtype=int))
    m_hat, j_hat, path = select_smallest_protective_param(np.arange(1, K + 1, dtype=int), losses_cal, alpha=ALPHA)
    path["trial"] = trial
    path["method"] = "crc"

    m_test = np.full(len(ranks_test), int(m_hat), dtype=int)
    losses_test = (ranks_test > m_test).astype(np.float32)

    return {
        "method": "crc",
        "selected_param": int(m_hat),
        "selected_m_mean": float(m_hat),
        "test_risk": float(np.mean(losses_test)),
        "avg_set_size": float(m_hat),
        "median_set_size": float(m_hat),
        "losses_test": losses_test,
        "sizes_test": m_test.astype(float),
        "calibration_path": path,
    }


def summarize_bins(method_result: Dict, bins: np.ndarray, trial: int):
    rows = []
    losses = method_result["losses_test"]
    sizes = method_result["sizes_test"]
    for b in np.unique(bins):
        mask = bins == b
        rows.append({
            "trial": trial,
            "method": method_result["method"],
            "bin": int(b),
            "n": int(mask.sum()),
            "bin_risk": float(np.mean(losses[mask])),
            "avg_set_size": float(np.mean(sizes[mask])),
            "median_set_size": float(np.median(sizes[mask])),
        })
    return rows


def add_summary_row(method_result: Dict, trial: int, base_acc: float, test_acc: float):
    return {
        "trial": trial,
        "method": method_result["method"],
        "alpha": ALPHA,
        "selected_param": method_result["selected_param"],
        "selected_m_mean": method_result["selected_m_mean"],
        "test_risk": method_result["test_risk"],
        "excess_risk_event": float(method_result["test_risk"] > ALPHA),
        "avg_set_size": method_result["avg_set_size"],
        "median_set_size": method_result["median_set_size"],
        "base_train_accuracy": base_acc,
        "base_test_accuracy": test_acc,
    }


# -----------------------------------------------------------------------------
# Single trial loop
# -----------------------------------------------------------------------------


def run_one_trial(trial: int, seed: int, X: np.ndarray, y: np.ndarray, K: int, risk_model: str = RISK_MODEL):
    idx_base, idx_risk, idx_cal, idx_test = make_four_way_split(y, seed)

    clf, proba = fit_base_classifier(X, y, idx_base, seed=seed, K=K)
    y_pred = proba.argmax(axis=1)
    base_acc = accuracy_score(y[idx_base], y_pred[idx_base])
    test_acc = accuracy_score(y[idx_test], y_pred[idx_test])

    ranks = true_label_ranks(proba, y)
    F = score_features(X, proba)

    F_risk, ranks_risk = F[idx_risk], ranks[idx_risk]
    F_cal, ranks_cal = F[idx_cal], ranks[idx_cal]
    F_test, ranks_test = F[idx_test], ranks[idx_test]

    difficulty = difficulty_score_from_proba(proba[idx_test])
    bins = make_equal_mass_bins(difficulty, n_bins=N_BINS)

    results = []
    paths = []

    res_crc = run_standard_crc(ranks_cal, ranks_test, trial, K)
    results.append(res_crc)
    paths.append(res_crc["calibration_path"])

    res_aa = run_aacrc_style(F_risk, ranks_risk, F_cal, ranks_cal, F_test, ranks_test, trial, K)
    results.append(res_aa)
    paths.append(res_aa["calibration_path"])

    res_recirc = run_recirc(F_risk, ranks_risk, F_cal, ranks_cal, F_test, ranks_test, trial, seed=seed + 1000, risk_model=risk_model)
    results.append(res_recirc)
    paths.append(res_recirc["calibration_path"])

    summary_rows = []
    bin_rows = []
    for res in results:
        summary_rows.append(add_summary_row(res, trial, base_acc, test_acc))
        bin_rows.extend(summarize_bins(res, bins, trial))

    return summary_rows, bin_rows, paths


# -----------------------------------------------------------------------------
# Aggregation and plots
# -----------------------------------------------------------------------------


def aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if len(summary_df) == 0:
        return pd.DataFrame()
    return (
        summary_df.groupby("method", as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            risk_mean=("test_risk", "mean"),
            risk_sd=("test_risk", "std"),
            excess_event_rate=("excess_risk_event", "mean"),
            avg_set_size_mean=("avg_set_size", "mean"),
            avg_set_size_sd=("avg_set_size", "std"),
            selected_m_mean=("selected_m_mean", "mean"),
            base_test_acc_mean=("base_test_accuracy", "mean"),
        )
        .sort_values("risk_mean")
    )


def aggregate_bins(bin_df: pd.DataFrame) -> pd.DataFrame:
    if len(bin_df) == 0:
        return pd.DataFrame()
    return (
        bin_df.groupby(["method", "bin"], as_index=False)
        .agg(
            bin_risk_mean=("bin_risk", "mean"),
            bin_risk_sd=("bin_risk", "std"),
            avg_set_size_mean=("avg_set_size", "mean"),
            n_mean=("n", "mean"),
        )
    )


def save_plots(summary_rows: List[Dict], bin_rows: List[Dict], output_dir: str, alpha: float = ALPHA):
    summary_df = pd.DataFrame(summary_rows)
    bin_df = pd.DataFrame(bin_rows)
    agg = aggregate_summary(summary_df)
    bins_agg = aggregate_bins(bin_df)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    for method, sub in summary_df.groupby("method"):
        sub = sub.sort_values("trial")
        plt.plot(sub["trial"], sub["test_risk"], marker="o", label=method)
    plt.axhline(alpha, linestyle="--", linewidth=1)
    plt.xlabel("Trial")
    plt.ylabel("Marginal test risk")
    plt.title("Risco marginal por trial")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marginal_risk_by_trial.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    methods = agg["method"].tolist()
    xpos = np.arange(len(methods))
    plt.bar(xpos, agg["risk_mean"].values)
    if len(agg) > 0:
        yerr = agg["risk_sd"].fillna(0).values / np.sqrt(np.maximum(agg["n_trials"].values, 1))
        plt.errorbar(xpos, agg["risk_mean"].values, yerr=yerr, fmt="none", capsize=4)
    plt.axhline(alpha, linestyle="--", linewidth=1)
    plt.xticks(xpos, methods, rotation=20, ha="right")
    plt.ylabel("Mean test risk")
    plt.title("Risco marginal médio acumulado")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marginal_risk_by_method.png"), dpi=160)
    plt.close()

    if len(bins_agg):
        plt.figure(figsize=(8, 4.5))
        for method, sub in bins_agg.groupby("method"):
            sub = sub.sort_values("bin")
            plt.plot(sub["bin"], sub["bin_risk_mean"], marker="o", label=method)
        plt.axhline(alpha, linestyle="--", linewidth=1)
        plt.xlabel("Difficulty bin")
        plt.ylabel("Mean bin risk")
        plt.title("Risco por bin de dificuldade")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bin_risk_curves.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        for method, sub in bins_agg.groupby("method"):
            sub = sub.sort_values("bin")
            plt.plot(sub["bin"], sub["avg_set_size_mean"], marker="o", label=method)
        plt.xlabel("Difficulty bin")
        plt.ylabel("Average set size")
        plt.title("Tamanho médio por bin de dificuldade")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bin_set_size_curves.png"), dpi=160)
        plt.close()


def build_final_summary(summary_df: pd.DataFrame, bin_df: pd.DataFrame):
    summary_by_method = (
        summary_df.groupby("method", as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            risk_mean=("test_risk", "mean"),
            risk_sd=("test_risk", "std"),
            risk_se=("test_risk", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            excess_event_rate=("excess_risk_event", "mean"),
            avg_set_size_mean=("avg_set_size", "mean"),
            avg_set_size_sd=("avg_set_size", "std"),
            selected_m_mean=("selected_m_mean", "mean"),
            selected_m_sd=("selected_m_mean", "std"),
            base_test_acc_mean=("base_test_accuracy", "mean"),
        )
    )

    worst_bins = (
        bin_df.groupby(["trial", "method"], as_index=False)
        .agg(
            worst_bin_risk=("bin_risk", "max"),
            q90_bin_risk=("bin_risk", lambda x: np.quantile(x, 0.90)),
            q95_bin_risk=("bin_risk", lambda x: np.quantile(x, 0.95)),
            mean_positive_excess_bin=("bin_risk", lambda x: np.mean(np.maximum(np.asarray(x) - ALPHA, 0))),
        )
    )

    worst_by_method = (
        worst_bins.groupby("method", as_index=False)
        .agg(
            worst_bin_mean=("worst_bin_risk", "mean"),
            worst_bin_sd=("worst_bin_risk", "std"),
            q90_bin_mean=("q90_bin_risk", "mean"),
            q95_bin_mean=("q95_bin_risk", "mean"),
            mean_positive_excess_bin=("mean_positive_excess_bin", "mean"),
        )
    )

    summary_by_method = summary_by_method.merge(worst_by_method, on="method", how="left")
    summary_by_method = summary_by_method.sort_values("risk_mean")

    bins_by_method = (
        bin_df.groupby(["method", "bin"], as_index=False)
        .agg(
            bin_risk_mean=("bin_risk", "mean"),
            bin_risk_sd=("bin_risk", "std"),
            avg_set_size_mean=("avg_set_size", "mean"),
            avg_set_size_sd=("avg_set_size", "std"),
            n_mean=("n", "mean"),
        )
    )

    compact_table = summary_by_method.copy()
    compact_table["marginal_risk"] = compact_table.apply(
        lambda r: f"{r['risk_mean']:.4f} ± {0 if pd.isna(r['risk_sd']) else r['risk_sd']:.4f}", axis=1
    )
    compact_table["worst_bin_risk"] = compact_table.apply(
        lambda r: f"{r['worst_bin_mean']:.4f} ± {0 if pd.isna(r['worst_bin_sd']) else r['worst_bin_sd']:.4f}", axis=1
    )
    compact_table["avg_set_size"] = compact_table.apply(
        lambda r: f"{r['avg_set_size_mean']:.2f} ± {0 if pd.isna(r['avg_set_size_sd']) else r['avg_set_size_sd']:.2f}", axis=1
    )
    compact_table = compact_table[[
        "method", "n_trials", "marginal_risk", "excess_event_rate", "worst_bin_risk", "avg_set_size"
    ]]

    return summary_by_method, bins_by_method, compact_table


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Experimento 5: Letter Recognition — CRC vs AA-CRC-style vs ReCIRC.")
    parser.add_argument("--output-dir", type=str, default=None, help="Diretório para salvar resultados.")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Número de trials do experimento.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Seed base para randomização.")
    parser.add_argument("--risk-model", type=str, default=RISK_MODEL, choices=["tabicl", "hgb"], help="Modelo para ReCIRC.")
    parser.add_argument("--no-plots", action="store_true", help="Desabilita geração de gráficos.")
    return parser.parse_args()


def main():
    args = parse_args()

    global N_TRIALS, BASE_SEED, RISK_MODEL, OUT_DIR, K
    N_TRIALS = args.trials
    BASE_SEED = args.seed
    RISK_MODEL = args.risk_model

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, CLASS_NAMES, raw_df = load_letter_recognition()
    N, P = X.shape
    K = len(CLASS_NAMES)

    print(f"N={N}, p={P}, K={K}")
    print("Classes:", CLASS_NAMES)

    all_summary_rows = []
    all_bin_rows = []
    all_calibration_paths = []

    for trial in tqdm(range(N_TRIALS), desc="Trials"):
        seed = BASE_SEED + 100 * trial
        summary_rows, bin_rows, paths = run_one_trial(trial=trial, seed=seed, X=X, y=y, K=K, risk_model=RISK_MODEL)
        all_summary_rows.extend(summary_rows)
        all_bin_rows.extend(bin_rows)
        all_calibration_paths.extend(paths)

        pd.DataFrame(all_summary_rows).to_csv(output_dir / "summary_results_incremental.csv", index=False)
        pd.DataFrame(all_bin_rows).to_csv(output_dir / "bin_results_incremental.csv", index=False)

    results_df = pd.DataFrame(all_summary_rows)
    bin_results_df = pd.DataFrame(all_bin_rows)
    if len(all_calibration_paths):
        calibration_paths_df = pd.concat(all_calibration_paths, ignore_index=True)
    else:
        calibration_paths_df = pd.DataFrame()

    results_df.to_csv(output_dir / "summary_results.csv", index=False)
    bin_results_df.to_csv(output_dir / "bin_results.csv", index=False)
    calibration_paths_df.to_csv(output_dir / "calibration_paths.csv", index=False)

    summary_by_method, bins_by_method, compact_table = build_final_summary(results_df, bin_results_df)
    summary_by_method.to_csv(output_dir / "summary_by_method_all_trials.csv", index=False)
    bins_by_method.to_csv(output_dir / "bins_by_method_all_trials.csv", index=False)
    compact_table.to_csv(output_dir / "compact_summary_all_trials.csv", index=False)

    print("\nResumo por método:")
    print(summary_by_method.round(4).to_string(index=False))
    print("\nTabela compacta:")
    print(compact_table.to_string(index=False))

    if not args.no_plots:
        save_plots(all_summary_rows, all_bin_rows, str(output_dir))
        print(f"\nGráficos salvos em: {output_dir}")

    meta = {
        "alpha": float(ALPHA),
        "n_trials": int(N_TRIALS),
        "seed": int(BASE_SEED),
        "risk_model": str(RISK_MODEL),
        "n_classes": int(K),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
