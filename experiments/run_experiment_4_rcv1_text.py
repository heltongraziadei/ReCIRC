#!/usr/bin/env python
"""RCV1 experiment (multilabel text) — CRC vs AA-CRC vs ReCIRC-TabICL.

Reuses the COCO multilabel pipeline (classes-as-items) but with RCV1 data.
The "items" are the 103 RCV1-v2 topics; prediction set is
C_lambda(x) = {k: p_hat_k(x) >= lambda} and the loss is the FNR = 1 - recall.

Base model: One-vs-Rest logistic regression on RCV1 TF-IDF — out-of-sample scores,
no GPU or transformer required. The conformal guarantee is model-agnostic.

Difficulty score is label-free (entropy of scores) for binning; fair budget
(FAIR_BUDGET) used for global CRC only; AA-CRC standardizes on D and fits
theta on C with an intercept-free ridge; B=1, alpha=0.10; documents with sum_k y_k == 0 are dropped.

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


AACRC_COMMIT = "64504c011ac2db910e258037e48170a63381b5e6"
AACRC_SOURCE_SHA256 = "9b3cffcb45f2e9a74f467ec94a00be7fcf2a24ec85f2dfe56f01d7fbd4a51315"

# --aacrc-repo is read here, before the import, because J/J_prime are loaded
# at module level. The full parser in parse_args() accepts the same flag.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--aacrc-repo", type=Path, default=None)
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.aacrc_repo is not None:
    AACRC_REPO = _pre_args.aacrc_repo.resolve()
else:
    AACRC_REPO = Path("AA-CRC").resolve()
    if not AACRC_REPO.exists():
        sh(f"git clone -q https://github.com/vincentblot28/AA-CRC.git {AACRC_REPO}")
        sh(f"git -C {AACRC_REPO} checkout -q {AACRC_COMMIT}")

_aacrc_source = AACRC_REPO / "multiaccurate_cp" / "utils" / "multiaccurate.py"
if not _aacrc_source.is_file():
    raise FileNotFoundError(
        f"AA-CRC source missing: {_aacrc_source}. Clone vincentblot28/AA-CRC, "
        f"checkout {AACRC_COMMIT}, and pass --aacrc-repo PATH."
    )
import hashlib
_digest = hashlib.sha256(_aacrc_source.read_bytes()).hexdigest()
if _digest != AACRC_SOURCE_SHA256:
    raise RuntimeError(
        f"Unexpected AA-CRC source SHA-256: {_digest}; expected {AACRC_SOURCE_SHA256} "
        f"(commit {AACRC_COMMIT})."
    )
sys.path.insert(0, str(AACRC_REPO))

try:
    from tabicl import TabICLRegressor  # noqa
except Exception:
    sh(f"{sys.executable} -m pip install -q tabicl")

# Importar AA-CRC oficial (falha explícita: o experimento não roda sem o baseline)
from multiaccurate_cp.utils.multiaccurate import J, J_prime
HAS_AACRC = True

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
USE_SYNTHETIC = False  # Mude para False para usar RCV1 real
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
    D_scores, cal_scores, cal_labels, test_scores, alpha, lambda_ridge=0.01,
    include_full_scores=True, seed=0, maxiter=1000
):
    """AA-CRC with the authors' J/J_prime (pinned commit).

    Same protocol as the other experiments: features standardized with D
    statistics; theta fitted on the calibration split C only; the ridge
    penalty excludes the intercept, so by Theorem 1 of Blot et al. the
    marginal level is not shifted by -2*rho*theta_0.
    """
    if not HAS_AACRC:
        return None

    feat = lambda s: make_features(s, include_full_scores=include_full_scores).values
    scaler = StandardScaler().fit(feat(D_scores))
    Phi_cal_b = np.column_stack(
        [np.ones(len(cal_scores)), scaler.transform(feat(cal_scores))]
    ).astype(np.float64)
    Phi_test_b = np.column_stack(
        [np.ones(len(test_scores)), scaler.transform(feat(test_scores))]
    ).astype(np.float64)

    n, D = Phi_cal_b.shape
    Y_aa = cal_labels[:, :, None].astype(np.float64)
    P_aa = cal_scores[:, :, None].astype(np.float64)

    mask = np.ones(D, dtype=np.float64)
    mask[0] = 0.0  # intercept not penalized

    def objective(theta):
        value = J(theta, Y_aa, P_aa, Phi_cal_b, alpha, n, None, None)
        return float(value + lambda_ridge * np.sum(mask * theta ** 2))

    def gradient(theta):
        grad = np.asarray(J_prime(theta, Y_aa, P_aa, Phi_cal_b, alpha, n, None, None), dtype=np.float64)
        return grad + 2.0 * lambda_ridge * mask * theta

    theta = np.zeros(D)
    theta[0] = 0.5

    t0 = time.time()
    res, messages = None, []
    for iterations in (maxiter, 3 * maxiter):
        res = minimize(objective, theta, method="SLSQP", jac=gradient,
                       options={"maxiter": iterations, "disp": False}, tol=1e-10)
        messages.append(str(res.message))
        if res.success and np.isfinite(res.x).all() and np.isfinite(res.fun):
            break
        if np.isfinite(res.x).all():
            theta = res.x.copy()
    else:
        raise RuntimeError("AA-CRC falhou: " + " | ".join(messages))
    dt = time.time() - t0

    lam_test = np.clip(Phi_test_b @ res.x, 0.0, 1.0)
    pred = prediction_sets(test_scores, lam_test)

    return {"pred": pred, "lambdas": lam_test, "theta": res.x, "time": dt,
            "attempts": len(messages)}


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


PREDICT_BATCH_ROWS = 40_000


def predict_risk_matrix(model, scores, lambda_grid, include_full_scores=True, enforce_monotone=True):
    """Prediz a matriz R(lambda|x) em lotes, para limitar a memória do TabICL."""
    base = make_features(scores, include_full_scores=include_full_scores).values.astype(np.float32)
    n, M = scores.shape[0], len(lambda_grid)
    rows_per_obs = max(1, PREDICT_BATCH_ROWS // M)
    lam_col = lambda_grid.astype(np.float32)
    R = np.empty((n, M), dtype=np.float64)
    for start in range(0, n, rows_per_obs):
        stop = min(n, start + rows_per_obs)
        m = stop - start
        X = np.concatenate(
            [np.repeat(base[start:stop], M, 0), np.tile(lam_col, m).reshape(-1, 1)], 1
        )
        R[start:stop] = np.clip(model.predict(X), 0, 1).reshape(m, M)
        del X

    if enforce_monotone:
        R = np.maximum.accumulate(R, axis=1)

    return R


def invert_risk_curve(R, lambda_grid, a):
    """Inverte R para obter lambda_a(x) tal que R(lambda_a(x)|x) <= a."""
    n_valid = (R <= a).sum(axis=1)
    lambdas = lambda_grid[np.maximum(n_valid - 1, 0)]
    return np.where(R[:, 0] > a, lambda_grid[0], lambdas)


def free_memory():
    import gc
    gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass


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
    elapsed = time.time() - t0

    del model, X_aug, Z, R_cal, R_test
    free_memory()

    return {
        "pred": pred,
        "lambdas": lam_test,
        "a_hat": a_hat,
        "risks_cal": risks_cal,
        "time": elapsed,
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
        o = run_aacrc(D_scores, cal_scores, cal_labels, test_scores, ALPHA, LAMBDA_RIDGE,
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
        o = run_aacrc(Ds, cs, cl, ts, ALPHA, LAMBDA_RIDGE, INCLUDE_FULL_SCORES, seed=seed)
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


def run_multiple_trials(sgmd, labels, n_trials=N_TRIALS, checkpoint_dir=None):
    """Executa múltiplos trials; cada trial é salvo em disco e pulado se já existir."""
    marg_list, cond_list, adapt_list = [], [], []
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for t in tqdm(range(n_trials), desc="Trials"):
        paths = None
        if checkpoint_dir is not None:
            paths = {k: checkpoint_dir / f"trial_{t:03d}_{k}.csv" for k in ("marg", "cond", "adapt")}
            if paths["marg"].exists() and paths["cond"].exists():
                m = pd.read_csv(paths["marg"])
                c = pd.read_csv(paths["cond"])
                a = pd.read_csv(paths["adapt"]) if paths["adapt"].exists() and paths["adapt"].stat().st_size > 1 else pd.DataFrame()
                print(f"Trial {t}: carregado do checkpoint.")
                marg_list.append(m); cond_list.append(c); adapt_list.append(a)
                continue

        m, c, a = run_one_split_trial(sgmd, labels, BASE_SEED + t)
        for d in (m, c, a):
            if len(d) > 0:
                d["trial"] = t
        if paths is not None:
            m.to_csv(paths["marg"], index=False)
            c.to_csv(paths["cond"], index=False)
            a.to_csv(paths["adapt"], index=False)
        marg_list.append(m)
        cond_list.append(c)
        adapt_list.append(a)
        free_memory()

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
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Número de trials.")
    parser.add_argument("--skip-single-split", action="store_true",
                        help="Pula o split único de sanity check (economiza uma rodada completa).")
    parser.add_argument("--predict-batch-rows", type=int, default=PREDICT_BATCH_ROWS,
                        help="Linhas por chamada de predict do TabICL (reduza se faltar memória).")
    parser.add_argument("--aacrc-repo", type=Path, default=None,
                        help=f"Repositório AA-CRC no commit {AACRC_COMMIT[:7]} (lido antes do import).")
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================


def main(args=None):
    global DEVICE, N_TRIALS, PREDICT_BATCH_ROWS
    if args is None:
        args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials deve ser positivo.")
    N_TRIALS = int(args.trials)
    PREDICT_BATCH_ROWS = int(args.predict_batch_rows)
    if args.device != "auto":
        if args.device == "cuda" and DEVICE != "cuda":
            raise RuntimeError("CUDA solicitada, mas não está disponível.")
        DEVICE = args.device
    print(f"Device: {DEVICE} | AA-CRC: {AACRC_REPO} | trials: {N_TRIALS}")
    print("=" * 80)
    print("Experimento 4: RCV1 (texto multilabel)")
    print("=" * 80)
    print()

    # Carrega dados
    print("Carregando dados...")
    sgmd, labels = load_data()
    print()

    # Checkpoints ficam fora da pasta com timestamp, para permitir retomada
    if args.output_dir is not None:
        checkpoint_dir = Path(args.output_dir) / "checkpoints"
    elif os.path.isdir("/content/drive"):
        checkpoint_dir = Path("/content/drive/MyDrive/PythonReCIRC/results/experiment_4_rcv1_text_checkpoints")
    else:
        checkpoint_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / "experiment_4_rcv1_text_checkpoints"
    print(f"Checkpoints: {checkpoint_dir}")

    # Split único com sanity check
    test_bins, test_labels = None, None
    if not args.skip_single_split:
        print("Executando split único...")
        results, test_bins, test_labels = run_single_split(sgmd, labels)
        free_memory()
        print()

    # Múltiplos trials
    print("Executando múltiplos trials...")
    df_marginal, df_conditional, df_adapt = run_multiple_trials(
        sgmd, labels, n_trials=N_TRIALS, checkpoint_dir=checkpoint_dir
    )
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
        "AACRC_COMMIT": AACRC_COMMIT,
        "AACRC_PROTOCOL": "features standardized on D; theta on C; ridge on slopes only",
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
