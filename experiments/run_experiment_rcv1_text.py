#!/usr/bin/env python
"""RCV1 experiment (multilabel text) — CRC vs AA-CRC vs ReCIRC-TabICL.

Reuses the COCO multilabel pipeline (classes-as-items) but with RCV1 data.
The "items" are the 103 RCV1-v2 topics; prediction set is
C_lambda(x) = {k: p_hat_k(x) >= lambda} and the loss is the FNR = 1 - recall.

Base model: One-vs-Rest logistic regression on RCV1 TF-IDF — out-of-sample scores,
no GPU or transformer required. The conformal guarantee is model-agnostic.

Difficulty score is label-free (entropy of scores) for binning; fair budget
(FAIR_BUDGET) used; B=1, alpha=0.10; documents with sum_k y_k == 0 are dropped.

By default the script runs a synthetic surrogate (cardinality ~3, heterogeneous
difficulty) to validate the end-to-end pipeline. Set USE_SYNTHETIC=False to
download the real RCV1 via scikit-learn.
"""


# Install required packages automatically
def ensure_packages():
    """Install required packages if they are not available."""
    import subprocess, sys
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
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_packages()

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import json


# ============================================================================
# 1. SETUP: clone do AA-CRC oficial e imports
# ============================================================================

def sh(c):
    print("$", c)
    return os.system(c)


if not os.path.exists("AA-CRC"):
    sh("git clone -q https://github.com/vincentblot28/AA-CRC.git")

sys.path.insert(0, "AA-CRC")

try:
    from tabicl import TabICLRegressor  # noqa
except Exception:
    sh(f"{sys.executable} -m pip install -q tabicl")

# Importar AA-CRC oficial
try:
    from multiaccurate_cp.utils.multiaccurate import J, J_prime
    HAS_AACRC = True
except Exception as e:
    HAS_AACRC = False
    print(f"AA-CRC indisponível ({e}).")

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from tabicl import TabICLRegressor
    HAS_TABICL = True
except Exception as e:
    print(f"TabICL indisponível ({e}); usando HistGradientBoostingRegressor.")
    HAS_TABICL = False

from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

print(f"Device: {DEVICE} | AA-CRC: {HAS_AACRC} | TabICL: {HAS_TABICL}")

plt.rcParams.update({"figure.dpi": 110})


# ============================================================================
# 2. CONFIGURAÇÃO
# ============================================================================

ALPHA = 0.10
B = 1.0
N_D = 2000
N_CAL = 1500
N_BINS = 10
DIFFICULTY_KIND = "entropy"

LAMBDA_GRID = np.linspace(0.0, 1.0, 26)
A_GRID = np.linspace(0.0, 0.5, 101)

LAMBDA_GRID_CRC = np.unique(np.concatenate([
    np.linspace(0.0, 0.25, 251),
    np.linspace(0.25, 1.0, 31),
]))

INCLUDE_FULL_SCORES = True
N_D_TABICL = 2000
LAMBDA_RIDGE = 0.01
FAIR_BUDGET = True
USE_AACRC = HAS_AACRC

# Fonte de dados
USE_SYNTHETIC = True  # Mude para False para usar RCV1 real
RCV1_N_BASE = 12000
RCV1_N_POOL = 18000

N_TRIALS = 20
BASE_SEED = 12345

# Cache dos escores OVR e rótulos
if os.environ.get("RCV1_CACHE_DIR"):
    RCV1_CACHE_DIR = os.environ["RCV1_CACHE_DIR"]
elif os.path.isdir("/content/drive"):
    RCV1_CACHE_DIR = "/content/drive/MyDrive/PythonReCIRC/results"
else:
    RCV1_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rcv1_cache")

# Diretório raiz para resultados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir("/content/drive"):
    RESULTS_ROOT = "/content/drive/MyDrive/PythonReCIRC/results"
else:
    RESULTS_ROOT = os.path.join(BASE_DIR, "results")


# ============================================================================
# 3. CARREGAMENTO DOS ESCORES E RÓTULOS
# ============================================================================

def make_synthetic_multilabel(n, K=80, seed=0):
    """Gera dados sintéticos multilabel com cardinalidade heterogênea."""
    r = np.random.default_rng(seed)
    diff = r.uniform(0, 1, n)
    Y = np.zeros((n, K), np.int8)
    S = np.zeros((n, K), np.float32)

    for i in range(n):
        npos = int(r.integers(1, 6))
        pos = r.choice(K, npos, replace=False)
        Y[i, pos] = 1
        base = r.uniform(0, 0.20, K)
        sharp = 0.55 + 0.35 * (1 - diff[i])
        base[pos] = np.clip(r.normal(sharp, 0.15 + 0.15 * diff[i], npos), 0, 1)
        S[i] = np.clip(base + r.normal(0, 0.05, K), 0, 1)

    return S, Y


def load_rcv1_multilabel(n_base=RCV1_N_BASE, n_pool=RCV1_N_POOL, seed=BASE_SEED):
    """RCV1-v2 via sklearn: treina OVR logística no split base, gera escores no pool conformal."""
    from sklearn.datasets import fetch_rcv1
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    rcv1 = fetch_rcv1()
    X, Y = rcv1.data, rcv1.target
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], n_base + n_pool, replace=False)
    Xb, Yb = X[idx[:n_base]], Y[idx[:n_base]].toarray()
    Xp, Yp = X[idx[n_base:]], Y[idx[n_base:]].toarray()

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200, C=1.0, solver="liblinear"), n_jobs=-1
    ).fit(Xb, Yb)

    sgmd = np.clip(clf.predict_proba(Xp), 1e-6, 1 - 1e-6).astype(np.float32)
    return sgmd, Yp.astype(np.int8)


def rcv1_cache_path(n_base=RCV1_N_BASE, n_pool=RCV1_N_POOL, seed=BASE_SEED):
    return os.path.join(RCV1_CACHE_DIR, f"rcv1_base{n_base}_pool{n_pool}_seed{seed}.npz")


def prepare_rcv1_cache_dir():
    """Monta o Google Drive (se disponível) e prepara o diretório de cache."""
    if RCV1_CACHE_DIR.startswith("/content/drive/") and not os.path.isdir("/content/drive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            print("Google Colab não disponível; usando RCV1_CACHE_DIR como definido.")

    os.makedirs(RCV1_CACHE_DIR, exist_ok=True)


def load_data():
    """Carrega ou gera os dados (sintéticos ou RCV1 real)."""
    if USE_SYNTHETIC:
        sgmd, labels = make_synthetic_multilabel(20000, K=80, seed=BASE_SEED)
    else:
        prepare_rcv1_cache_dir()
        CACHE = rcv1_cache_path()

        if os.path.isfile(CACHE):
            with np.load(CACHE) as z:
                sgmd = z["sgmd"].astype(np.float32)
                labels = z["labels"].astype(np.int8)
            expected_shape = (RCV1_N_POOL, 103)
            if sgmd.shape != expected_shape or labels.shape != expected_shape:
                raise ValueError(
                    f"Cache incompatível: sgmd={sgmd.shape}, labels={labels.shape}; "
                    f"esperado {expected_shape}."
                )
            print(f"✓ cache carregado: {CACHE}")
        else:
            print(f"Cache não encontrado; treinando OVR e salvando em: {CACHE}")
            sgmd, labels = load_rcv1_multilabel()
            np.savez_compressed(CACHE, sgmd=sgmd.astype(np.float32), labels=labels.astype(np.int8))
            print(f"✓ cache salvo: {CACHE}")

    keep = labels.sum(1) > 0
    sgmd, labels = sgmd[keep], labels[keep]
    N_CLASSES = sgmd.shape[1]
    print(
        f"sgmd: {sgmd.shape} | labels: {labels.shape} | classes: {N_CLASSES} "
        f"| cardinalidade média: {labels.sum(1).mean():.2f}"
    )

    return sgmd, labels


# ============================================================================
# 4. FUNÇÕES BÁSICAS: perdas, features, bins de dificuldade
# ============================================================================


def prediction_sets(scores, lam):
    """Constrói conjuntos de predição S_lambda(x)."""
    if np.isscalar(lam):
        return scores >= lam
    return scores >= lam[:, None]


def per_image_fnr(pred_set, gt_labels):
    """FNR por imagem: (1 - recall)."""
    denom = gt_labels.sum(axis=1).clip(min=1)
    recall_i = (pred_set * gt_labels).sum(axis=1) / denom
    return 1.0 - recall_i


def avg_fnr(pred_set, gt_labels):
    """FNR médio."""
    return float(per_image_fnr(pred_set, gt_labels).mean())


def evaluate_pred_set(pred_set, gt_labels):
    """Avalia um conjunto de predição."""
    return {
        "fnr": avg_fnr(pred_set, gt_labels),
        "avg_size": float(pred_set.sum(axis=1).mean()),
    }


# ============================================================================
# 5. FEATURES
# ============================================================================


SUMMARY_COLS = [
    "max_score", "second_score", "third_score", "sum_scores", "mean_score",
    "sd_score", "entropy", "n_above_005", "n_above_010", "n_above_020", "n_above_050"
]


def summary_features(scores):
    """Extrai features de resumo dos escores."""
    eps = 1e-12
    s_sorted = np.sort(scores, axis=1)[:, ::-1]
    entropy = -np.sum(scores * np.log(scores + eps) + (1 - scores) * np.log(1 - scores + eps), axis=1)

    return pd.DataFrame({
        "max_score": s_sorted[:, 0],
        "second_score": s_sorted[:, 1],
        "third_score": s_sorted[:, 2],
        "sum_scores": scores.sum(1),
        "mean_score": scores.mean(1),
        "sd_score": scores.std(1),
        "entropy": entropy,
        "n_above_005": (scores >= 0.05).sum(1),
        "n_above_010": (scores >= 0.10).sum(1),
        "n_above_020": (scores >= 0.20).sum(1),
        "n_above_050": (scores >= 0.50).sum(1),
    })


def make_features(scores, include_full_scores=True):
    """Monta matriz de features."""
    X = summary_features(scores)
    if include_full_scores:
        cols = pd.DataFrame(scores, columns=[f"score_{j:03d}" for j in range(scores.shape[1])])
        X = pd.concat([X.reset_index(drop=True), cols.reset_index(drop=True)], axis=1)
    return X


# ============================================================================
# 6. DIFICULDADE E BINS
# ============================================================================


def difficulty_score(scores, kind="entropy"):
    """Calcula score de dificuldade."""
    X = summary_features(scores)
    if kind == "entropy":
        return X["entropy"].to_numpy()
    if kind == "neg_max_score":
        return -X["max_score"].to_numpy()
    raise ValueError(kind)


def fit_difficulty_edges(scores, n_bins=4, kind="entropy"):
    """Ajusta as arestas dos bins de dificuldade."""
    diff = difficulty_score(scores, kind)
    edges = np.unique(np.quantile(diff, np.linspace(0, 1, n_bins + 1))).astype(float)
    if len(edges) < n_bins + 1:
        edges = np.linspace(diff.min(), diff.max(), n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def apply_difficulty_bins(scores, edges, kind="entropy"):
    """Aplica bins de dificuldade."""
    diff = difficulty_score(scores, kind)
    return pd.cut(diff, bins=edges, labels=False, include_lowest=True).astype(int)


# ============================================================================
# 7. CRC MARGINAL (baseline)
# ============================================================================


def risk_curve(scores, gt, lambda_grid):
    """Calcula a curva de risco para um grid de lambdas."""
    return np.array([avg_fnr(prediction_sets(scores, lam), gt) for lam in lambda_grid])


def choose_lambda_crc(scores, gt, alpha, lambda_grid, B=1.0):
    """CRC marginal: escolhe lambda pelo bound de Hoeffding."""
    n = scores.shape[0]
    risks = risk_curve(scores, gt, lambda_grid)
    bound = (n / (n + 1)) * risks + B / (n + 1)
    valid = np.where(bound <= alpha)[0]
    return float(lambda_grid[valid[-1]]) if len(valid) else float(lambda_grid[0])


# ============================================================================
# 8. AA-CRC (Blot et al., 2025)
# ============================================================================


def run_aacrc(
    cal_scores, cal_labels, test_scores, alpha, lambda_ridge=0.01,
    include_full_scores=True, seed=0
):
    """Roda AA-CRC com J/J_prime oficiais."""
    if not HAS_AACRC:
        return None

    n = cal_scores.shape[0]
    feat = lambda s: make_features(s, include_full_scores=include_full_scores).values
    scaler = StandardScaler().fit(feat(cal_scores))
    Phi_cal = scaler.transform(feat(cal_scores))
    Phi_test = scaler.transform(feat(test_scores))

    Phi_cal_b = np.concatenate([np.ones((len(Phi_cal), 1)), Phi_cal], 1).astype(np.float64)
    Phi_test_b = np.concatenate([np.ones((len(Phi_test), 1)), Phi_test], 1).astype(np.float64)

    D = Phi_cal_b.shape[1]
    Y_aa = cal_labels[:, :, None].astype(np.float64)
    P_aa = cal_scores[:, :, None].astype(np.float64)

    theta0 = np.zeros(D)
    theta0[0] = 0.5

    t0 = time.time()
    res = minimize(
        J,
        x0=theta0,
        method="SLSQP",
        args=(Y_aa, P_aa, Phi_cal_b, alpha, n, "ridge", lambda_ridge),
        jac=J_prime,
        options={"maxiter": 200, "disp": False},
        tol=1e-6,
    )
    dt = time.time() - t0

    lam_test = np.clip(Phi_test_b @ res.x, 0.0, 1.0)
    pred = prediction_sets(test_scores, lam_test)

    return {"pred": pred, "lambdas": lam_test, "theta": res.x, "time": dt}


# ============================================================================
# 9. ReCIRC-TabICL (Rota 2)
# ============================================================================


def build_augmented(scores, gt, lambda_grid, include_full_scores=True):
    """Constrói dataset aumentado (phi(X_i), lambda_m) -> Z = loss."""
    base = make_features(scores, include_full_scores=include_full_scores).values
    n, M = scores.shape[0], len(lambda_grid)
    Z = np.stack([per_image_fnr(prediction_sets(scores, lam), gt) for lam in lambda_grid], 1)
    X_aug = np.concatenate(
        [np.repeat(base, M, 0), np.tile(lambda_grid, n).reshape(-1, 1)], 1
    )
    return X_aug.astype(np.float32), Z.reshape(-1)


def fit_risk_model(X_aug, Z, device="cpu", seed=0):
    """Ajusta o modelo de risco (TabICL ou HistGradientBoosting)."""
    if HAS_TABICL:
        m = TabICLRegressor(n_estimators=4, device=device, random_state=seed)
    else:
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
    m.fit(X_aug, Z)
    return m


def predict_risk_matrix(model, scores, lambda_grid, include_full_scores=True, enforce_monotone=True):
    """Prediz a matriz R(lambda|x)."""
    base = make_features(scores, include_full_scores=include_full_scores).values
    n, M = scores.shape[0], len(lambda_grid)
    X = np.concatenate(
        [np.repeat(base, M, 0), np.tile(lambda_grid, n).reshape(-1, 1)], 1
    ).astype(np.float32)
    R = np.clip(model.predict(X), 0, 1).reshape(n, M)

    if enforce_monotone:
        R = np.maximum.accumulate(R, axis=1)

    return R


def invert_risk_curve(R, lambda_grid, a):
    """Inverte R para obter lambda_a(x) tal que R(lambda_a(x)|x) <= a."""
    n_valid = (R <= a).sum(axis=1)
    lambdas = lambda_grid[np.maximum(n_valid - 1, 0)]
    return np.where(R[:, 0] > a, lambda_grid[0], lambdas)


def run_recirc(
    D_scores, D_labels, cal_scores, cal_labels, test_scores, lambda_grid, a_grid, alpha,
    include_full_scores=True, n_d_tabicl=1000, device="cpu", seed=0
):
    """Roda ReCIRC-TabICL."""
    rng = np.random.default_rng(seed)

    if HAS_TABICL and len(D_scores) > n_d_tabicl:
        idx = rng.choice(len(D_scores), n_d_tabicl, replace=False)
        D_sc, D_lb = D_scores[idx], D_labels[idx]
    else:
        D_sc, D_lb = D_scores, D_labels

    t0 = time.time()

    # Treina o modelo de risco
    X_aug, Z = build_augmented(D_sc, D_lb, lambda_grid, include_full_scores)
    model = fit_risk_model(X_aug, Z, device=device, seed=seed)

    # Prediz e calibra
    R_cal = predict_risk_matrix(model, cal_scores, lambda_grid, include_full_scores)
    R_test = predict_risk_matrix(model, test_scores, lambda_grid, include_full_scores)

    risks_cal = np.array(
        [
            per_image_fnr(
                prediction_sets(cal_scores, invert_risk_curve(R_cal, lambda_grid, a)),
                cal_labels,
            ).mean()
            for a in a_grid
        ]
    )

    n_cal = len(cal_scores)
    bound = (n_cal / (n_cal + 1)) * risks_cal + B / (n_cal + 1)
    valid = np.where(bound <= alpha)[0]
    a_hat = float(a_grid[valid[-1]]) if len(valid) else float(a_grid[0])

    lam_test = invert_risk_curve(R_test, lambda_grid, a_hat)
    pred = prediction_sets(test_scores, lam_test)

    return {
        "pred": pred,
        "lambdas": lam_test,
        "a_hat": a_hat,
        "risks_cal": risks_cal,
        "time": time.time() - t0,
    }


# ============================================================================
# 10. EXECUÇÃO: Split único com sanity check
# ============================================================================


def run_single_split(sgmd, labels):
    """Executa um split único e compara CRC vs AA-CRC vs ReCIRC."""
    rng = np.random.default_rng(BASE_SEED)
    perm = rng.permutation(len(sgmd))

    D_idx = perm[:N_D]
    cal_idx = perm[N_D : N_D + N_CAL]
    test_idx = perm[N_D + N_CAL :]

    D_scores, D_labels = sgmd[D_idx], labels[D_idx]
    cal_scores, cal_labels = sgmd[cal_idx], labels[cal_idx]
    test_scores, test_labels = sgmd[test_idx], labels[test_idx]

    if FAIR_BUDGET:
        calM_scores = np.concatenate([D_scores, cal_scores])
        calM_labels = np.concatenate([D_labels, cal_labels])
    else:
        calM_scores, calM_labels = cal_scores, cal_labels

    bin_edges = fit_difficulty_edges(D_scores, N_BINS, DIFFICULTY_KIND)
    test_bins = apply_difficulty_bins(test_scores, bin_edges, DIFFICULTY_KIND)

    print(
        f"N={len(sgmd)} | D={len(D_idx)} | cal={len(cal_idx)} | test={len(test_idx)} "
        f"| calM={len(calM_scores)}"
    )

    results = {}

    # CRC marginal
    lam_crc = choose_lambda_crc(calM_scores, calM_labels, ALPHA, LAMBDA_GRID_CRC, B=B)
    pred_crc = prediction_sets(test_scores, lam_crc)
    results["CRC marginal"] = {
        "pred": pred_crc,
        "lambdas": np.full(len(test_scores), lam_crc),
        **evaluate_pred_set(pred_crc, test_labels),
    }
    print(
        f"CRC marginal : lam={lam_crc:.3f} | "
        f"FNR={results['CRC marginal']['fnr']:.4f} | "
        f"tam={results['CRC marginal']['avg_size']:.2f}"
    )

    # AA-CRC
    if USE_AACRC:
        o = run_aacrc(calM_scores, calM_labels, test_scores, ALPHA, LAMBDA_RIDGE,
                      INCLUDE_FULL_SCORES, seed=0)
        if o:
            results["AA-CRC"] = {**o, **evaluate_pred_set(o["pred"], test_labels)}
            print(
                f"AA-CRC       : FNR={results['AA-CRC']['fnr']:.4f} | "
                f"tam={results['AA-CRC']['avg_size']:.2f} | "
                f"lam={o['lambdas'].mean():.3f}±{o['lambdas'].std():.3f} | {o['time']:.1f}s"
            )

    # ReCIRC
    rk = "ReCIRC-TabICL" if HAS_TABICL else "ReCIRC-HGB"
    o = run_recirc(
        D_scores, D_labels, cal_scores, cal_labels, test_scores, LAMBDA_GRID, A_GRID,
        ALPHA, INCLUDE_FULL_SCORES, n_d_tabicl=N_D_TABICL, device=DEVICE, seed=0
    )
    results[rk] = {**o, **evaluate_pred_set(o["pred"], test_labels)}
    print(
        f"{rk:13s}: a_hat={o['a_hat']:.3f} | "
        f"FNR={results[rk]['fnr']:.4f} | tam={results[rk]['avg_size']:.2f} | "
        f"lam={o['lambdas'].mean():.3f}±{o['lambdas'].std():.3f} | {o['time']:.1f}s"
    )

    # Tabela de resumo
    df = pd.DataFrame(
        [
            {
                "método": k,
                "FNR": r["fnr"],
                "tam_médio": r["avg_size"],
                "lambda_médio": float(np.mean(r["lambdas"])),
                "lambda_sd": float(np.std(r["lambdas"])),
            }
            for k, r in results.items()
        ]
    )
    print(df.round(4).to_string(index=False))

    se = np.sqrt(ALPHA * (1 - ALPHA) / len(test_labels))
    print(f"\nFNR deve cair perto de alpha={ALPHA} (±2SE=±{2*se:.4f}).")
    for k, r in results.items():
        print(f"  {k:14s} z=({r['fnr']:.4f}-{ALPHA})/SE = {(r['fnr']-ALPHA)/se:+.2f}")

    return results, test_bins, test_labels


# ============================================================================
# 11. MÚLTIPLOS SPLITS: variância e cobertura condicional
# ============================================================================


def conditional_by_bin(pred, gt, bins, n_bins, name):
    """Calcula risco condicional por bin de dificuldade."""
    loss_i = per_image_fnr(pred, gt)
    size_i = pred.sum(1)
    return pd.DataFrame(
        [
            {
                "method": name,
                "bin": b,
                "n": int((bins == b).sum()),
                "conditional_risk": float(loss_i[bins == b].mean()) if (bins == b).any() else np.nan,
                "mean_set_size": float(size_i[bins == b].mean()) if (bins == b).any() else np.nan,
            }
            for b in range(n_bins)
        ]
    )


def run_one_split_trial(scores, gt, seed):
    """Executa um trial do experimento de múltiplos splits."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(scores))
    D_i, c_i, t_i = perm[:N_D], perm[N_D : N_D + N_CAL], perm[N_D + N_CAL :]

    Ds, Dl = scores[D_i], gt[D_i]
    cs, cl = scores[c_i], gt[c_i]
    ts, tl = scores[t_i], gt[t_i]

    cMs, cMl = (np.concatenate([Ds, cs]), np.concatenate([Dl, cl])) if FAIR_BUDGET else (cs, cl)

    edges = fit_difficulty_edges(Ds, N_BINS, DIFFICULTY_KIND)
    tb = apply_difficulty_bins(ts, edges, DIFFICULTY_KIND)

    preds, lam_img = {}, {}

    # CRC marginal
    lam = choose_lambda_crc(cMs, cMl, ALPHA, LAMBDA_GRID_CRC, B=B)
    preds["CRC marginal"] = prediction_sets(ts, lam)
    lam_img["CRC marginal"] = None

    # AA-CRC
    if USE_AACRC:
        o = run_aacrc(cMs, cMl, ts, ALPHA, LAMBDA_RIDGE, INCLUDE_FULL_SCORES, seed=seed)
        if o:
            preds["AA-CRC"] = o["pred"]
            lam_img["AA-CRC"] = o["lambdas"]

    # ReCIRC
    rk = "ReCIRC-TabICL" if HAS_TABICL else "ReCIRC-HGB"
    o = run_recirc(
        Ds, Dl, cs, cl, ts, LAMBDA_GRID, A_GRID, ALPHA, INCLUDE_FULL_SCORES,
        n_d_tabicl=N_D_TABICL, device=DEVICE, seed=seed
    )
    preds[rk] = o["pred"]
    lam_img[rk] = o["lambdas"]

    # Resumos marginais
    marg = pd.DataFrame(
        [
            {"method": k, "test_fnr": avg_fnr(p, tl), "avg_size": float(p.sum(1).mean())}
            for k, p in preds.items()
        ]
    )

    # Resumos condicionais
    cond = pd.concat(
        [conditional_by_bin(p, tl, tb, N_BINS, k) for k, p in preds.items()],
        ignore_index=True,
    )

    # Lambdas por bin
    adapt_rows = []
    for k in preds:
        if lam_img[k] is not None:
            for b in range(N_BINS):
                if (tb == b).any():
                    adapt_rows.append(
                        {
                            "method": k,
                            "bin": b,
                            "mean_lambda": float(np.mean(lam_img[k][tb == b])),
                            "sd_lambda": float(np.std(lam_img[k][tb == b])),
                        }
                    )
                else:
                    adapt_rows.append(
                        {"method": k, "bin": b, "mean_lambda": np.nan, "sd_lambda": np.nan}
                    )

    adapt = pd.DataFrame(adapt_rows) if adapt_rows else pd.DataFrame()

    return marg, cond, adapt


def run_multiple_trials(sgmd, labels, n_trials=N_TRIALS):
    """Executa múltiplos trials e agrega resultados."""
    marg_list, cond_list, adapt_list = [], [], []

    for t in tqdm(range(n_trials), desc="Trials"):
        m, c, a = run_one_split_trial(sgmd, labels, BASE_SEED + t)
        for d in (m, c, a):
            if len(d) > 0:
                d["trial"] = t
        marg_list.append(m)
        cond_list.append(c)
        adapt_list.append(a)

    df_marginal = pd.concat(marg_list, ignore_index=True)
    df_conditional = pd.concat(cond_list, ignore_index=True)
    df_adapt = pd.concat(adapt_list, ignore_index=True) if any(len(a) for a in adapt_list) else pd.DataFrame()

    print(f"Concluído: {n_trials} trials.")
    return df_marginal, df_conditional, df_adapt


# ============================================================================
# 12. RESUMOS AGREGADOS E VISUALIZAÇÕES
# ============================================================================


def aggregate_results(df_marginal, df_conditional, df_adapt):
    """Agrega e sumariza os resultados."""
    summary_marginal = df_marginal.groupby("method", as_index=False).agg(
        mean_test_fnr=("test_fnr", "mean"),
        sd_test_fnr=("test_fnr", "std"),
        mean_avg_size=("avg_size", "mean"),
        violation_rate=("test_fnr", lambda x: float(np.mean(x > ALPHA))),
    )

    cond = df_conditional.copy()
    cond["excess"] = np.maximum(cond["conditional_risk"] - ALPHA, 0.0)

    per_trial = cond.groupby(["trial", "method"], as_index=False).agg(
        worst_bin_risk=("conditional_risk", "max"), mean_excess_by_bin=("excess", "mean")
    )

    summary_conditional = per_trial.groupby("method", as_index=False).agg(
        mean_worst_bin=("worst_bin_risk", "mean"), mean_excess=("mean_excess_by_bin", "mean")
    )

    summary = summary_marginal.merge(summary_conditional, on="method", how="left")
    order = ["CRC marginal", "AA-CRC", "ReCIRC-TabICL", "ReCIRC-HGB"]
    summary["__o"] = summary["method"].apply(lambda m: order.index(m) if m in order else 99)
    summary = summary.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("RESUMO AGREGADO")
    print("=" * 80)
    print(summary.round(4).to_string(index=False))

    return summary, per_trial


def plot_results(df_marginal, per_trial, test_labels):
    """Plota os resultados principais."""
    methods_order = [
        m for m in ["CRC marginal", "AA-CRC", "ReCIRC-TabICL", "ReCIRC-HGB"]
        if m in df_marginal["method"].unique()
    ]
    palette = {"CRC marginal": "#1f77b4", "AA-CRC": "#ff7f0e", "ReCIRC-TabICL": "#d62728", "ReCIRC-HGB": "#d62728"}

    final = df_marginal.merge(per_trial, on=["trial", "method"], how="left")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    panels = [
        ("test_fnr", r"$\widehat R_{test}$", "(a) Risco marginal", True),
        ("worst_bin_risk", r"$\max_b \widehat R_b$", "(b) Pior bin", True),
        ("mean_excess_by_bin", "Excesso médio", "(c) Excesso por bin", False),
        ("avg_size", "Tamanho do conjunto", "(d) Tamanho médio", False),
    ]

    jit = np.random.default_rng(0)
    for ax, (metric, ylab, title, show_a) in zip(axes, panels):
        for j, m in enumerate(methods_order):
            tmp = final[final["method"] == m]
            x = np.full(len(tmp), j, float) + jit.normal(0, 0.04, len(tmp))
            ax.scatter(x, tmp[metric], alpha=0.75, s=45, color=palette.get(m, "gray"))
            ax.hlines(tmp[metric].mean(), j - 0.2, j + 0.2, lw=2.5, color="black")

        if show_a:
            ax.axhline(ALPHA, ls="--", color="gray", lw=1.5)

        ax.set_xticks(range(len(methods_order)))
        ax.set_xticklabels(methods_order, fontsize=9, rotation=15)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle(
        f"RCV1 texto multilabel | alpha={ALPHA} | {final['trial'].nunique()} splits",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_conditional_coverage(df_conditional):
    """Plota cobertura condicional por bin."""
    methods_order = [
        m for m in ["CRC marginal", "AA-CRC", "ReCIRC-TabICL", "ReCIRC-HGB"]
        if m in df_conditional["method"].unique()
    ]
    palette = {"CRC marginal": "#1f77b4", "AA-CRC": "#ff7f0e", "ReCIRC-TabICL": "#d62728", "ReCIRC-HGB": "#d62728"}

    agg = df_conditional.groupby(["method", "bin"], as_index=False).agg(
        mean_cond_risk=("conditional_risk", "mean"),
        sd_cond_risk=("conditional_risk", "std"),
        mean_size=("mean_set_size", "mean"),
    )

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
            color=palette.get(m, "gray"),
        )
        axes[1].plot(tmp["bin"], tmp["mean_size"], marker="o", label=m, color=palette.get(m, "gray"))

    axes[0].axhline(ALPHA, ls="--", color="gray", label=fr"$\alpha={ALPHA}$")
    axes[0].set_xlabel("Bin de dificuldade (entropia)")
    axes[0].set_ylabel("FNR condicional")
    axes[0].set_title("Cobertura condicional")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Bin de dificuldade")
    axes[1].set_ylabel("Tamanho médio")
    axes[1].set_title("Tamanho por bin")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# ARGUMENTOS
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Experimento RCV1 texto multilabel.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório para salvar resultados. Se não fornecido, tenta /content/drive (Colab) ou local.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo para TabICL: auto (CUDA se disponível), cpu, ou cuda.",
    )
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main(args=None):
    if args is None:
        args = parse_args()
    print("=" * 80)
    print("Experimento 4: RCV1 (texto multilabel)")
    print("=" * 80)
    print()

    # Carrega dados
    print("Carregando dados...")
    sgmd, labels = load_data()
    print()

    # Split único com sanity check
    print("Executando split único...")
    results, test_bins, test_labels = run_single_split(sgmd, labels)
    print()

    # Múltiplos trials
    print("Executando múltiplos trials...")
    df_marginal, df_conditional, df_adapt = run_multiple_trials(sgmd, labels, n_trials=N_TRIALS)
    print()

    # Agregação
    summary, per_trial = aggregate_results(df_marginal, df_conditional, df_adapt)
    print()

    # Diretório de resultados
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        results_dir = str(Path(args.output_dir) / f"experiment_4_rcv1_text_{timestamp}")
    elif os.path.isdir("/content/drive"):
        results_dir = os.path.join("/content/drive/MyDrive/PythonReCIRC/results", f"experiment_4_rcv1_text_{timestamp}")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, "results", f"experiment_4_rcv1_text_{timestamp}")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSalvando resultados em: {results_dir}")

    # Salvar DataFrames
    df_marginal.to_csv(os.path.join(results_dir, "df_marginal.csv"), index=False)
    df_conditional.to_csv(os.path.join(results_dir, "df_conditional.csv"), index=False)
    df_adapt.to_csv(os.path.join(results_dir, "df_adapt.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    per_trial.to_csv(os.path.join(results_dir, "per_trial.csv"), index=False)

    # Salvar arrays/objetos auxiliares
    try:
        np.savez_compressed(os.path.join(results_dir, "extras.npz"), test_bins=test_bins, test_labels=test_labels)
    except Exception:
        pass

    # Salvar metadados
    meta = {
        "timestamp": timestamp,
        "alpha": float(ALPHA),
        "B": float(B),
        "N_D": int(N_D),
        "N_CAL": int(N_CAL),
        "N_TRIALS": int(N_TRIALS),
        "USE_SYNTHETIC": bool(USE_SYNTHETIC),
    }
    with open(os.path.join(results_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Visualizações: gerar, salvar e fechar
    print("Gerando e salvando figuras...")
    try:
        fig = plot_results(df_marginal, per_trial, test_labels)
        fig.savefig(os.path.join(results_dir, "figure_marginal.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("Falha ao salvar figura marginal:", e)

    try:
        fig2 = plot_conditional_coverage(df_conditional)
        fig2.savefig(os.path.join(results_dir, "figure_conditional.png"), bbox_inches="tight")
        plt.close(fig2)
    except Exception as e:
        print("Falha ao salvar figura condicional:", e)

    print("\n" + "=" * 80)
    print("Experimento concluído!")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
