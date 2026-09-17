#!/usr/bin/env python3

"""Synthetic multilabel experiment runner with simple and complex DGPs.

The script compares global CRC, AA-CRC and ReCIRC.  The main AA-CRC baseline
uses the random-forest leaf feature map: D is used to learn a task-adaptive
partition and theta is fitted only on the independent calibration split C.

Scenarios
---------
``simple`` reproduces the original one-dimensional latent-difficulty design.

``complex`` is a pre-specified two-dimensional smooth stress test with
X=(X1,X2) ~ Unif([0,1]^2).  The latent difficulty is additive and nonlinear:
it increases linearly with X1 and smoothly with sin^2(pi X2).  Difficulty changes
both label cardinality and score quality.  The true latent difficulty is used
only to define evaluation bins; it is never supplied directly to CRC, AA-CRC
or ReCIRC.

The complex scenario is intended to test adaptation under multidimensional
heterogeneity without introducing interactions.  Both adaptive methods receive
the observed covariates and the same score summaries; AA-CRC uses them to learn
RF leaves, whereas ReCIRC uses them jointly with the decision parameter to
estimate the conditional-risk surface.
"""

# Install required packages automatically
def ensure_packages():
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
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabicl import TabICLRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
AACRC_CANDIDATES = [
    Path(os.environ["AACRC_DIR"]) if "AACRC_DIR" in os.environ else None,
    SCRIPT_DIR / "AA-CRC",
    SCRIPT_DIR.parent / "AA-CRC",
    Path.cwd() / "AA-CRC",
    Path("/content/recirc/AA-CRC"),
    Path("/content/AA-CRC"),
]
AACRC_DIR = next(
    (
        path
        for path in AACRC_CANDIDATES
        if path is not None
        and (path / "multiaccurate_cp" / "utils" / "multiaccurate.py").is_file()
    ),
    None,
)
AACRC_COMMIT = "64504c011ac2db910e258037e48170a63381b5e6"
if AACRC_DIR is not None:
    sys.path.insert(0, str(AACRC_DIR))

try:
    from multiaccurate_cp.utils.multiaccurate import J, J_prime
except ImportError as exc:
    searched = ", ".join(str(path) for path in AACRC_CANDIDATES if path is not None)
    raise ImportError(f"AA-CRC não encontrado. Caminhos verificados: {searched}") from exc


B_LOSS = 1.0
AACRC_RIDGE = 0.01
AACRC_MAXITER = 1000

METHOD_ORDER = []
METHOD_LABELS = {
    "crc": "Standard CRC",
    "aacrc_linear": "AA-CRC (linear)",
    "aacrc_rf": "AA-CRC (RF leaves)",
    "recirc": "ReCIRC TabICL",
    "oracle": "Oracle Rectified",
}
METHOD_COLORS = {
    "crc": "#999999",
    "aacrc_linear": "#009E73",
    "aacrc_rf": "#CC79A7",
    "recirc": "#002F6C",
    "oracle": "#D55E00",
}
RF_LEAF_NMIN_DIAG = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Experimento sintético multilabel.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dgp-seed", type=int, default=123)
    parser.add_argument(
        "--scenario", choices=["simple", "complex"], default="simple",
        help="Data-generating scenario. complex is a two-dimensional additive smooth stress test.",
    )
    parser.add_argument("--n-context", type=int, default=1000)
    parser.add_argument("--n-cal", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--labels", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--lambda-points", type=int, default=101)
    parser.add_argument("--a-points", type=int, default=101)
    parser.add_argument("--k-aug", type=int, default=20)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--n-mc-oracle", type=int, default=200)
    parser.add_argument("--skip-oracle", action="store_true")
    # AA-CRC feature-map choices. RF is the paper-style data-driven group map.
    parser.add_argument("--aacrc-basis", choices=["linear", "rf", "both"], default="linear")
    parser.add_argument(
        "--aacrc-ridge-intercept", action="store_true",
        help="Also penalize the intercept in the linear arm (authors' full ridge).",
    )
    parser.add_argument(
        "--aacrc-features", choices=["x", "feats"], default="feats",
        help="Covariates used to build AA-CRC maps: raw X only, or X plus score summaries.",
    )
    parser.add_argument("--rf-trees", type=int, default=3)
    parser.add_argument("--rf-depth", type=int, default=4)
    parser.add_argument("--rf-min-leaf", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def make_label_probs(labels, rng):
    popularity = rng.normal(0.0, 0.7, size=labels)
    probs = expit(popularity)
    return probs / probs.sum()


def raw_covariates(data):
    X = np.asarray(data["x"], dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X


def dgp_quantities(X, scenario):
    """Return latent difficulty, expected cardinality, separation and noise SD."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]

    if scenario == "simple":
        x1 = X[:, 0]
        difficulty = x1
        mean_relevant = 2.0 + 6.0 * x1
        separation = 3.0 * (1.0 - x1) + 0.6
        noise_sd = 0.5 + 1.3 * x1
        return difficulty, mean_relevant, separation, noise_sd

    if scenario != "complex":
        raise ValueError("scenario must be 'simple' or 'complex'.")
    if X.shape[1] != 2:
        raise ValueError("The complex scenario requires two covariates.")

    x1, x2 = X[:, 0], X[:, 1]

    # Simple two-dimensional additive difficulty:
    # one monotone component and one smooth nonlinear component.
    smooth_x2 = np.sin(np.pi * x2) ** 2
    difficulty = np.clip(0.10 + 0.50 * x1 + 0.35 * smooth_x2, 0.0, 1.0)

    # Harder inputs have more relevant labels and poorer score separation.
    mean_relevant = 2.0 + 5.0 * difficulty
    separation = 3.6 - 2.5 * difficulty
    noise_sd = 0.45 + 1.45 * difficulty
    return difficulty, mean_relevant, separation, noise_sd


def generate_data(n, labels, rng, label_probs=None, scenario="complex"):
    if scenario == "simple":
        X = rng.uniform(0.0, 1.0, size=(n, 1))
    else:
        X = rng.uniform(0.0, 1.0, size=(n, 2))

    difficulty, mean_relevant, separation, noise_sd = dgp_quantities(X, scenario)
    n_relevant = np.clip(rng.poisson(mean_relevant), 1, labels)

    if label_probs is None:
        label_probs = make_label_probs(labels, rng)

    y = np.zeros((n, labels), dtype=int)
    for i in range(n):
        positive = rng.choice(labels, size=n_relevant[i], replace=False, p=label_probs)
        y[i, positive] = 1

    logits = np.empty((n, labels))
    for i in range(n):
        logits[i] = (
            separation[i] * (2 * y[i] - 1)
            + rng.normal(0.0, noise_sd[i], labels)
        )

    return {
        "x": X,
        "difficulty": difficulty,
        "y": y,
        "scores": expit(logits),
        "n_relevant": n_relevant,
        "label_probs": label_probs,
    }


def concatenate(*splits):
    return {
        "x": np.concatenate([raw_covariates(data) for data in splits], axis=0),
        "difficulty": np.concatenate([data["difficulty"] for data in splits]),
        "y": np.concatenate([data["y"] for data in splits], axis=0),
        "scores": np.concatenate([data["scores"] for data in splits], axis=0),
        "n_relevant": np.concatenate([data["n_relevant"] for data in splits]),
        "label_probs": splits[0]["label_probs"],
    }


def prediction_set(scores, lam):
    return scores >= 1.0 - lam


def missed_positive_loss(scores, y, lam):
    selected = prediction_set(scores, lam)
    return 1.0 - (selected & y).sum(axis=1) / y.sum(axis=1)


def losses_over_lambda(scores, y, lambda_grid):
    losses = np.empty((len(scores), len(lambda_grid)))
    for j, lam in enumerate(lambda_grid):
        losses[:, j] = missed_positive_loss(scores, y, lam)
    return losses


def crc_select(losses, grid, alpha):
    upper = (len(losses) * losses.mean(axis=0) + B_LOSS) / (len(losses) + 1)
    valid = np.flatnonzero(upper <= alpha)
    return float(grid[valid[0]]) if len(valid) else float(grid[-1])


def score_summary(scores):
    ordered = np.sort(scores)
    return np.array(
        [
            scores.mean(),
            scores.std(),
            scores.max(),
            ordered[-5:].mean(),
            scores.sum(),
            ordered[-1] - ordered[-2],
        ]
    )


def recirc_base_features(data):
    X = raw_covariates(data)
    summaries = np.vstack([score_summary(s) for s in data["scores"]])
    return np.column_stack([X, summaries])


def build_augmented_context(context, lambda_grid, k_aug, rng):
    base = recirc_base_features(context)
    rows = []
    targets = []
    for i in range(len(base)):
        lambdas = rng.choice(lambda_grid, size=k_aug, replace=True)
        for lam in lambdas:
            loss = missed_positive_loss(
                context["scores"][i : i + 1], context["y"][i : i + 1], lam
            )[0]
            rows.append(np.concatenate([base[i], [lam]]))
            targets.append(loss)
    return np.asarray(rows, dtype=float), np.asarray(targets)


def risk_features(data, lambda_grid):
    base = recirc_base_features(data)
    rows = []
    indices = []
    for i in range(len(base)):
        for lam in lambda_grid:
            rows.append(np.concatenate([base[i], [lam]]))
            indices.append(i)
    return np.asarray(rows, dtype=float), np.asarray(indices)


def fit_tabicl_surface(context, lambda_grid, k_aug, device, seed, rng):
    features, target = build_augmented_context(context, lambda_grid, k_aug, rng)
    model = TabICLRegressor(device=device, kv_cache=True, random_state=seed)
    model.fit(features, target)
    return model


def estimate_tabicl_risk(model, data, lambda_grid):
    features, indices = risk_features(data, lambda_grid)
    predicted = np.clip(model.predict(features), 0.0, 1.0)
    risk = np.empty((len(data["x"]), len(lambda_grid)))
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for i in range(len(data["x"])):
        values = predicted[indices == i]
        risk[i] = np.clip(-iso.fit_transform(lambda_grid, -values), 0.0, 1.0)
    return risk


def invert_risk_matrix(risk, lambda_grid, a):
    mask = risk <= a
    first = np.argmax(mask, axis=1)
    first[~mask.any(axis=1)] = len(lambda_grid) - 1
    return lambda_grid[first]


def calibrate_rectified(cal, risk_cal, lambda_grid, a_grid, alpha):
    losses = np.empty((len(cal["x"]), len(a_grid)))
    for j, a in enumerate(a_grid):
        lam = invert_risk_matrix(risk_cal, lambda_grid, a)
        for i in range(len(cal["x"])):
            losses[i, j] = missed_positive_loss(
                cal["scores"][i : i + 1], cal["y"][i : i + 1], lam[i]
            )[0]
    upper = (len(cal["x"]) * losses.mean(axis=0) + B_LOSS) / (len(cal["x"]) + 1)
    valid = np.flatnonzero(upper <= alpha)
    return float(a_grid[valid[-1]]) if len(valid) else float(a_grid[0])


def aacrc_base_features(data, source="feats"):
    """Test-time-available covariates used before constructing the AA-CRC map."""
    X = raw_covariates(data)
    if source == "x":
        return X
    if source != "feats":
        raise ValueError("source must be 'x' or 'feats'.")
    summaries = np.vstack([score_summary(s) for s in data["scores"]])
    return np.column_stack([X, summaries]).astype(float)


def individual_target_lambda(data, alpha):
    """Smallest lambda giving per-sample missed-positive loss <= alpha.

    Since prediction_set = {score >= 1-lambda}, the highest admissible threshold
    is the m-th largest positive-label score, where m=ceil((1-alpha)*#positives).
    Its complementary lambda is the minimum protection required by that sample.
    """
    y = np.asarray(data["y"], dtype=bool)
    scores = np.asarray(data["scores"], dtype=float)
    target = np.empty(len(scores), dtype=float)
    for i in range(len(scores)):
        positive_scores = scores[i, y[i]]
        if positive_scores.size == 0:
            raise ValueError("AA-CRC RF target requires at least one positive label per sample.")
        needed = int(np.ceil((1.0 - alpha) * positive_scores.size - 1e-12))
        needed = min(max(needed, 1), positive_scores.size)
        threshold_star = np.sort(positive_scores)[::-1][needed - 1]
        # Numerical safeguard: lambda = 1 - threshold_star can reconstruct
        # 1 - lambda infinitesimally above threshold_star in floating point,
        # excluding the boundary positive label. Move lambda by one ULP toward 1.
        target[i] = np.nextafter(1.0 - threshold_star, 1.0)
    return np.clip(target, 0.0, 1.0)


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class LeafFeatureMap:
    """Phi(x): one-hot indicators of the leaves reached in each RF tree."""

    def __init__(self, data_res, alpha, source, n_trees, max_depth, min_leaf, seed):
        Z = aacrc_base_features(data_res, source)
        target = individual_target_lambda(data_res, alpha)
        self.rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            min_samples_split=min_leaf,
            n_jobs=-1,
            random_state=seed,
        ).fit(Z, target)
        self.encoder = _one_hot_encoder().fit(self.rf.apply(Z))
        self.n_trees = int(n_trees)
        self.source = source
        self.target_mean = float(target.mean())
        self.target_std = float(target.std(ddof=1)) if len(target) > 1 else 0.0

    def __call__(self, data):
        Z = aacrc_base_features(data, self.source)
        return self.encoder.transform(self.rf.apply(Z)).astype(np.float64)

    @property
    def n_leaves(self):
        return int(sum(len(c) for c in self.encoder.categories_))


def run_aacrc_optimizer(phi, labels, scores, alpha, theta0, *, ridge=None, ridge_mask=None,
                        bounds=None, name="AA-CRC"):
    """Authors' J / J_prime without internal regularization, plus an external ridge.

    With ridge_mask[0] = 0 the intercept is not penalized, so by Theorem 1 of
    Blot et al. the marginal level is not shifted (same convention as the XOR,
    Insurance and Superconductor scripts). A mask of ones recovers the authors'
    full ridge.
    """
    phi = np.asarray(phi, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    theta = np.asarray(theta0, dtype=np.float64).copy()
    attempts = []
    start = time.time()
    result = None
    valid = False
    rho = 0.0 if ridge is None else float(ridge)
    mask = np.zeros(phi.shape[1]) if ridge_mask is None else np.asarray(ridge_mask, dtype=np.float64)
    n = len(phi)

    def objective(t):
        return float(J(t, labels, scores, phi, alpha, n, None, None) + rho * np.sum(mask * t ** 2))

    def gradient(t):
        grad = np.asarray(J_prime(t, labels, scores, phi, alpha, n, None, None), dtype=np.float64)
        return grad + 2.0 * rho * mask * t

    for maxiter in (AACRC_MAXITER, 3 * AACRC_MAXITER):
        result = minimize(
            objective,
            theta,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            options={"maxiter": maxiter, "disp": False},
            tol=1e-10,
        )
        valid = bool(result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun))
        attempts.append(str(result.message))
        if valid:
            break
        if np.all(np.isfinite(result.x)):
            theta = result.x.copy()
    if result is None or not valid:
        raise RuntimeError(f"{name} falhou: " + " | ".join(attempts))
    return result, time.time() - start, len(attempts)


def fit_aacrc_linear(context, cal, alpha, source="feats", ridge_intercept=False):
    """Previous linear map, but theta is fitted on C only."""
    Z_D = aacrc_base_features(context, source)
    Z_C = aacrc_base_features(cal, source)
    scaler = StandardScaler().fit(Z_D)
    phi = scaler.transform(Z_C)
    phi = np.column_stack([np.ones(len(phi)), phi]).astype(np.float64)
    labels = cal["y"][:, :, None].astype(np.float64)
    scores = cal["scores"][:, :, None].astype(np.float64)
    theta0 = np.zeros(phi.shape[1], dtype=np.float64)
    theta0[0] = 0.5
    mask = np.ones(phi.shape[1], dtype=np.float64)
    if not ridge_intercept:
        mask[0] = 0.0
    result, elapsed, attempts = run_aacrc_optimizer(
        phi, labels, scores, alpha, theta0, ridge=AACRC_RIDGE, ridge_mask=mask,
        name="AA-CRC (linear)",
    )
    threshold = np.clip(phi @ result.x, 0.0, 1.0)
    fit_lambda = 1.0 - threshold
    return {
        "theta": result.x,
        "scaler": scaler,
        "source": source,
        "time": elapsed,
        "objective": float(result.fun),
        "attempts": attempts,
        "fit_risk": float(missed_positive_loss(cal["scores"], cal["y"], fit_lambda[:, None]).mean()),
        "n_params": int(phi.shape[1]),
        "ridge_on_intercept": bool(ridge_intercept),
    }


def predict_aacrc_linear(fit, data, lambda_grid):
    phi = fit["scaler"].transform(aacrc_base_features(data, fit["source"]))
    phi = np.column_stack([np.ones(len(phi)), phi])
    threshold = np.clip(phi @ fit["theta"], 0.0, 1.0)
    return np.clip(1.0 - threshold, lambda_grid[0], lambda_grid[-1])


def fit_aacrc_rf(cal, alpha, leaf_map, lambda_init):
    """RF-leaf AA-CRC: theta on C, no ridge, thresholds constrained to [0,1]."""
    phi = leaf_map(cal)
    labels = cal["y"][:, :, None].astype(np.float64)
    scores = cal["scores"][:, :, None].astype(np.float64)
    T = leaf_map.n_trees
    # Each sample activates one leaf per tree, hence sum_j phi_j theta_j is in
    # [0,1] whenever every leaf coefficient is in [0,1/T].
    lower, upper = 0.0, 1.0 / T
    threshold_init = float(np.clip(1.0 - lambda_init, 0.0, 1.0))
    theta0 = np.full(phi.shape[1], threshold_init / T, dtype=np.float64)
    bounds = [(lower, upper)] * phi.shape[1]
    result, elapsed, attempts = run_aacrc_optimizer(
        phi, labels, scores, alpha, theta0, ridge=None, bounds=bounds, name="AA-CRC (RF leaves)"
    )
    threshold = np.clip(phi @ result.x, 0.0, 1.0)
    fit_lambda = 1.0 - threshold
    fit_loss = missed_positive_loss(cal["scores"], cal["y"], fit_lambda[:, None])
    counts = phi.sum(axis=0)
    leaf_risk = (phi * fit_loss[:, None]).sum(axis=0) / np.maximum(counts, 1.0)
    populated = counts >= RF_LEAF_NMIN_DIAG
    at_bound = np.isclose(result.x, lower, atol=1e-9) | np.isclose(result.x, upper, atol=1e-9)
    return {
        "theta": result.x,
        "leaf_map": leaf_map,
        "time": elapsed,
        "objective": float(result.fun),
        "attempts": attempts,
        "fit_risk": float(fit_loss.mean()),
        "fit_worst_leaf_risk": float(leaf_risk[populated].max()) if populated.any() else float("nan"),
        "n_params": int(phi.shape[1]),
        "n_leaves": int(leaf_map.n_leaves),
        "rate_theta_at_bound": float(at_bound.mean()),
        "rf_target_mean": leaf_map.target_mean,
        "rf_target_std": leaf_map.target_std,
    }


def predict_aacrc_rf(fit, data, lambda_grid):
    phi = fit["leaf_map"](data)
    threshold = np.clip(phi @ fit["theta"], 0.0, 1.0)
    return np.clip(1.0 - threshold, lambda_grid[0], lambda_grid[-1])


def validate_aacrc_mapping(data):
    error = 0.0
    for threshold in np.linspace(0.0, 1.0, 21):
        selected = data["scores"] >= threshold
        aacrc_loss = 1.0 - (selected & data["y"]).sum(axis=1) / data["y"].sum(axis=1)
        experiment_loss = missed_positive_loss(data["scores"], data["y"], 1.0 - threshold)
        error = max(error, float(np.abs(aacrc_loss - experiment_loss).max()))
    if error > 1e-12:
        raise AssertionError(f"Mapeamento AA-CRC inválido: {error:.3e}")
    return error


def validate_rf_target(data, alpha):
    """Check that the RF target is minimally protective on the supplied samples."""
    lam = individual_target_lambda(data, alpha)
    loss = missed_positive_loss(data["scores"], data["y"], lam[:, None])
    if np.any(loss > alpha + 1e-12):
        raise AssertionError("RF target failed to attain the per-sample target loss.")
    # One infinitesimal decrease in lambda should violate the target unless a tie
    # in positive scores creates a flat boundary; this is diagnostic only.
    return float(loss.max()), float(loss.mean())

def replicate_at_x(x, label_probs, n_mc, labels, rng, scenario):
    x = np.asarray(x, dtype=float).reshape(1, -1)
    _, mean_relevant, separation, noise_sd = dgp_quantities(x, scenario)
    mean_relevant = float(mean_relevant[0])
    separation = float(separation[0])
    noise_sd = float(noise_sd[0])

    n_relevant = np.clip(rng.poisson(mean_relevant, size=n_mc), 1, labels)
    y = np.zeros((n_mc, labels), dtype=int)
    for i in range(n_mc):
        positive = rng.choice(labels, size=n_relevant[i], replace=False, p=label_probs)
        y[i, positive] = 1
    logits = separation * (2 * y - 1) + rng.normal(0.0, noise_sd, size=(n_mc, labels))
    return y, expit(logits)


def oracle_risk_matrix(data, lambda_grid, label_probs, n_mc, labels, rng, scenario):
    X = raw_covariates(data)
    risk = np.empty((len(X), len(lambda_grid)))
    for i, x in enumerate(X):
        y, scores = replicate_at_x(x, label_probs, n_mc, labels, rng, scenario)
        risk[i] = losses_over_lambda(scores, y, lambda_grid).mean(axis=0)
    return risk


def evaluate(data, lam, alpha, method, n_bins, lambda_grid):
    loss = np.empty(len(data["x"]))
    size = np.empty(len(data["x"]))
    for i in range(len(data["x"])):
        loss[i] = missed_positive_loss(
            data["scores"][i : i + 1], data["y"][i : i + 1], lam[i]
        )[0]
        size[i] = prediction_set(data["scores"][i : i + 1], lam[i]).sum()
    bins = pd.qcut(data["difficulty"], q=n_bins, labels=False, duplicates="drop")
    groups = np.sort(pd.Series(bins).dropna().unique())
    bin_risk = np.array([loss[bins == group].mean() for group in groups])
    bin_size = np.array([size[bins == group].mean() for group in groups])
    row = {
        "method": method,
        "risk": loss.mean(),
        "worst_bin_risk": bin_risk.max(),
        "worst_bin_ratio": bin_risk.max() / alpha,
        "mean_bin_excess": np.maximum(bin_risk - alpha, 0.0).mean(),
        "mean_set_size": size.mean(),
        "lambda_mean": lam.mean(),
        "lambda_sd": lam.std(ddof=1),
        "lambda_min_rate": np.isclose(lam, lambda_grid[0]).mean(),
        "lambda_max_rate": np.isclose(lam, lambda_grid[-1]).mean(),
    }
    return row, bin_risk, bin_size


try:
    from .risk_calibration import RiskCalibration, bin_groups
except ImportError:
    from risk_calibration import RiskCalibration, bin_groups


def run_trial(args, seed, device, label_probs, diagnostic=None):
    rng = np.random.default_rng(seed)
    lambda_grid = np.linspace(0.0, 1.0, args.lambda_points)
    a_grid = np.linspace(0.0, 1.0, args.a_points)
    context = generate_data(args.n_context, args.labels, rng, label_probs, args.scenario)
    cal = generate_data(args.n_cal, args.labels, rng, label_probs, args.scenario)
    test = generate_data(args.n_test, args.labels, rng, label_probs, args.scenario)
    fit_data = concatenate(context, cal)

    crc_losses = losses_over_lambda(fit_data["scores"], fit_data["y"], lambda_grid)
    lambda_crc = crc_select(crc_losses, lambda_grid, args.alpha)
    lambdas = {"crc": np.full(args.n_test, lambda_crc)}
    diagnostics = {"lambda_crc": lambda_crc}

    if args.aacrc_basis in ("linear", "both"):
        fit_linear = fit_aacrc_linear(
            context, cal, args.alpha, source=args.aacrc_features,
            ridge_intercept=args.aacrc_ridge_intercept,
        )
        lambdas["aacrc_linear"] = predict_aacrc_linear(fit_linear, test, lambda_grid)
        diagnostics.update({
            f"aacrc_linear_{k}": v
            for k, v in fit_linear.items()
            if k not in ("theta", "scaler")
        })

    if args.aacrc_basis in ("rf", "both"):
        leaf_map = LeafFeatureMap(
            context,
            args.alpha,
            args.aacrc_features,
            args.rf_trees,
            args.rf_depth,
            args.rf_min_leaf,
            seed,
        )
        fit_rf = fit_aacrc_rf(cal, args.alpha, leaf_map, lambda_crc)
        lambdas["aacrc_rf"] = predict_aacrc_rf(fit_rf, test, lambda_grid)
        diagnostics.update({
            f"aacrc_rf_{k}": v
            for k, v in fit_rf.items()
            if k not in ("theta", "leaf_map")
        })

    tabicl = fit_tabicl_surface(context, lambda_grid, args.k_aug, device, seed, rng)
    recirc_risk_cal = estimate_tabicl_risk(tabicl, cal, lambda_grid)
    recirc_risk_test = estimate_tabicl_risk(tabicl, test, lambda_grid)
    a_recirc = calibrate_rectified(cal, recirc_risk_cal, lambda_grid, a_grid, args.alpha)
    lambdas["recirc"] = invert_risk_matrix(recirc_risk_test, lambda_grid, a_recirc)

    a_oracle = np.nan
    if not args.skip_oracle:
        oracle_rng = np.random.default_rng(seed + 9999)
        oracle_risk_cal = oracle_risk_matrix(
            cal, lambda_grid, label_probs, args.n_mc_oracle, args.labels, oracle_rng, args.scenario
        )
        oracle_risk_test = oracle_risk_matrix(
            test, lambda_grid, label_probs, args.n_mc_oracle, args.labels, oracle_rng, args.scenario
        )
        a_oracle = calibrate_rectified(cal, oracle_risk_cal, lambda_grid, a_grid, args.alpha)
        lambdas["oracle"] = invert_risk_matrix(oracle_risk_test, lambda_grid, a_oracle)

    diagnostics.update({"a_recirc": a_recirc, "a_oracle": a_oracle})

    rows = []
    bin_rows = []
    for method in METHOD_ORDER:
        if method not in lambdas:
            continue
        lam = lambdas[method]
        row, bin_risk, bin_size = evaluate(
            test, lam, args.alpha, method, args.bins, lambda_grid
        )
        rows.append(row)
        for group, (risk, size) in enumerate(zip(bin_risk, bin_size), start=1):
            bin_rows.append({"method": method, "bin": group, "risk": risk, "set_size": size})

    if diagnostic is not None:
        bins = np.asarray(pd.qcut(test["difficulty"], q=args.bins, labels=False, duplicates="drop"))
        losses_by_budget = np.column_stack([
            missed_positive_loss(
                test["scores"], test["y"],
                invert_risk_matrix(recirc_risk_test, lambda_grid, a)[:, None]
            )
            for a in a_grid
        ])
        groups = bin_groups(bins, args.bins)
        diagnostic.add_trial(a_grid, losses_by_budget, groups, seed=seed, a_hat=a_recirc)

    return pd.DataFrame(rows), pd.DataFrame(bin_rows), diagnostics

def summarize(data, metrics):
    summary = data.groupby("method", as_index=False).agg(
        n_trials=("trial", "nunique"),
        **{
            metric: (metric, "mean")
            for metric in metrics
        },
        **{
            f"{metric}_sd": (metric, "std")
            for metric in metrics
        },
    )
    for metric in metrics:
        summary[f"{metric}_se"] = summary[f"{metric}_sd"] / np.sqrt(summary["n_trials"])
        summary[f"{metric}_ci95_low"] = summary[metric] - 1.96 * summary[f"{metric}_se"]
        summary[f"{metric}_ci95_high"] = summary[metric] + 1.96 * summary[f"{metric}_se"]
    return summary.sort_values("worst_bin_risk")


def paired_difference(data, first, second, metric):
    paired = data.pivot(index="trial", columns="method", values=metric).dropna()
    difference = paired[first] - paired[second]
    se = difference.std(ddof=1) / np.sqrt(len(difference))
    return {
        "comparison": f"{METHOD_LABELS[first]} - {METHOD_LABELS[second]}",
        "metric": metric,
        "n_pairs": len(difference),
        "mean_diff": difference.mean(),
        "sd_diff": difference.std(ddof=1),
        "ci95_low": difference.mean() - 1.96 * se,
        "ci95_high": difference.mean() + 1.96 * se,
        "first_better_fraction": (difference < 0.0).mean(),
    }


def plot_results(results, bin_results, output_dir, alpha):
    methods = [method for method in METHOD_ORDER if method in set(results["method"])]
    metrics = [
        ("risk", "Marginal risk"),
        ("worst_bin_risk", "Worst-bin risk"),
        ("mean_set_size", "Mean set size"),
    ]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        for position, method in enumerate(methods):
            values = results.loc[results["method"] == method, metric].to_numpy()
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(position + jitter, values, color=METHOD_COLORS[method], alpha=0.7)
            ax.hlines(values.mean(), position - 0.25, position + 0.25, color="black", lw=2)
        if metric in {"risk", "worst_bin_risk"}:
            ax.axhline(alpha, color="gray", ls="--")
        ax.set_title(title)
        ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    grouped = bin_results.groupby(["method", "bin"])["risk"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        data = grouped[grouped["method"] == method]
        x = data["bin"].to_numpy()
        mean = data["mean"].to_numpy()
        sem = data["sem"].to_numpy()
        ax.plot(x, mean, "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=METHOD_COLORS[method], alpha=0.15)
    ax.axhline(alpha, color="gray", ls="--")
    ax.set(xlabel="Difficulty bin", ylabel="Risk")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "conditional_risk.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-found"


def main():
    global METHOD_ORDER
    args = parse_args()
    METHOD_ORDER = ["crc"]
    if args.aacrc_basis in ("linear", "both"):
        METHOD_ORDER.append("aacrc_linear")
    if args.aacrc_basis in ("rf", "both"):
        METHOD_ORDER.append("aacrc_rf")
    METHOD_ORDER.append("recirc")
    if not args.skip_oracle:
        METHOD_ORDER.append("oracle")
    if args.labels < 5:
        raise ValueError("--labels deve ser pelo menos 5.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas não está disponível.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "results" / f"multilabel_{args.scenario}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_rng = np.random.default_rng(args.dgp_seed)
    label_probs = make_label_probs(args.labels, label_rng)
    validation = generate_data(
        128,
        args.labels,
        np.random.default_rng(args.dgp_seed + 1),
        label_probs,
        args.scenario,
    )
    mapping_error = validate_aacrc_mapping(validation)
    rf_target_max_loss, rf_target_mean_loss = validate_rf_target(validation, args.alpha)
    print(f"Scenario: {args.scenario}")
    print(f"AA-CRC loss mapping max error: {mapping_error:.3e}")
    print(
        "AA-CRC RF target check: "
        f"max_loss={rf_target_max_loss:.6f}, mean_loss={rf_target_mean_loss:.6f}"
    )

    risk_diagnostic = RiskCalibration()
    all_metrics = []
    all_bins = []
    diagnostics = []
    start = time.time()
    for trial in range(args.trials):
        seed = args.seed + trial + 1
        trial_start = time.time()
        metrics, bins, diag = run_trial(args, seed, device, label_probs, diagnostic=risk_diagnostic)
        metrics["trial"] = trial
        metrics["seed"] = seed
        bins["trial"] = trial
        bins["seed"] = seed
        diag.update({"trial": trial, "seed": seed})
        all_metrics.append(metrics)
        all_bins.append(bins)
        diagnostics.append(diag)
        compact = metrics[["method", "risk", "worst_bin_risk", "mean_set_size"]]
        print(f"\nTrial {trial + 1}/{args.trials} ({time.time() - trial_start:.1f}s)")
        print(compact.round(4).to_string(index=False))

    risk_diagnostic.save(args.output_dir, make_plot=not args.no_plots)
    results = pd.concat(all_metrics, ignore_index=True)
    bin_results = pd.concat(all_bins, ignore_index=True)
    diagnostics_df = pd.DataFrame(diagnostics)
    metric_names = ["risk", "worst_bin_risk", "mean_bin_excess", "mean_set_size"]
    summary = summarize(results, metric_names)
    paired_rows = []
    for baseline in ("aacrc_rf", "aacrc_linear"):
        if baseline in set(results["method"]):
            paired_rows.extend(
                paired_difference(results, "recirc", baseline, metric)
                for metric in metric_names
            )
            paired_rows.append(
                paired_difference(results, baseline, "crc", "worst_bin_risk")
            )
    paired = pd.DataFrame(paired_rows)

    results.to_csv(args.output_dir / "per_replication_metrics.csv", index=False)
    bin_results.to_csv(args.output_dir / "per_replication_bin_metrics.csv", index=False)
    diagnostics_df.to_csv(args.output_dir / "per_replication_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)

    metadata = {
        "experiment": "synthetic_multilabel_classification",
        "scenario": args.scenario,
        "scenario_definition": (
            "simple: one-dimensional latent difficulty; "
            "complex: two-dimensional additive smooth difficulty, 0.10 + 0.50*x1 + 0.35*sin(pi*x2)^2"
        ),
        "protocol": {
            "crc": "D union C",
            "aacrc_linear": (
                "feature standardization on D; theta on C; ridge "
                + ("on all coefficients" if args.aacrc_ridge_intercept else "on slopes only (free intercept)")
            ),
            "aacrc_rf": (
                "RF on D predicts minimum per-sample lambda attaining FNR <= alpha; "
                "RF inputs are observed X plus score summaries when --aacrc-features=feats; "
                "Phi = RF leaf indicators; theta on C; no ridge; leaf coefficients bounded"
            ),
            "recirc": "risk surface on D using observed X plus score summaries and lambda; rectification budget on C",
        },
        "arguments": {**vars(args), "output_dir": str(args.output_dir), "device_resolved": device},
        "aacrc_commit": AACRC_COMMIT,
        "package_versions": {
            name: package_version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "tabicl", "torch"]
        },
    }
    with (args.output_dir / "protocol_config.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    if not args.no_plots:
        plot_results(results, bin_results, args.output_dir, args.alpha)

    print("\nSummary")
    print(summary.round(4).to_string(index=False))
    print("\nPaired differences")
    print(paired.round(4).to_string(index=False))
    print(f"\nTempo total: {time.time() - start:.1f}s")
    print(f"Resultados: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
