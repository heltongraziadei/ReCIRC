#!/usr/bin/env python3

"""Synthetic heteroscedastic experiment runner.

Generates heteroscedastic synthetic data, compares CRC / AA-CRC / ReCIRC and
saves results and plots.

AA-CRC arms (chosen with --aacrc-basis):

* ``aacrc_rf``   -- feature map of Blot et al. (2025, Sec. 3, Algorithm 1) for
  tabular regression: a random forest is trained on the absolute residuals of
  the context split D, and Phi(x) is the one-hot vector of the leaves reached
  by x (one group per leaf). theta is fitted on the calibration split C only,
  without ridge and with box bounds, as in the authors' tabular notebook
  (notebooks/cqr_example.ipynb, commit 64504c0).
* ``aacrc_poly`` -- previous polynomial basis {1, x, |x|, x^2}. By default it is
  also fitted on C only (--aacrc-poly-split union restores D u C) and its ridge
  penalty excludes the intercept (--poly-ridge-intercept restores the old
  behaviour). By Theorem 1 of Blot et al., a ridge on the intercept lowers the
  marginal level to alpha - 2*rho*theta_0.

Both arms use the authors' objective J / J_prime (multiaccurate.py) and the
pseudo-label reduction of the bounded excess loss (error <= 1/AACRC_N_PSEUDO).
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
        "pygam": "pygam",
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
from pygam import LinearGAM, s
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabicl import TabICLRegressor

try:
    from .risk_calibration import RiskCalibration, bin_groups
except ImportError:
    from risk_calibration import RiskCalibration, bin_groups


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
EXCESS_SCALE = 1.5
AACRC_N_PSEUDO = 101
AACRC_RIDGE = 1e-3
AACRC_MAXITER = 300

# Filled in main() according to --aacrc-basis.
METHOD_ORDER = []
METHOD_LABELS = {
    "crc": "Standard CRC",
    "aacrc_poly": "AA-CRC (poly)",
    "aacrc_rf": "AA-CRC (RF leaves)",
    "recirc": "ReCIRC TabICL",
    "oracle": "Oracle Rectified",
}
METHOD_COLORS = {
    "crc": "#999999",
    "aacrc_poly": "#009E73",
    "aacrc_rf": "#CC79A7",
    "recirc": "#002F6C",
    "oracle": "#D55E00",
}

SIGMA_SCENARIO = "abs"


def parse_args():
    parser = argparse.ArgumentParser(description="Experimento sintético heteroscedástico.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-context", type=int, default=1000)
    parser.add_argument("--n-cal", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--lambda-points", type=int, default=101)
    parser.add_argument("--a-points", type=int, default=101)
    parser.add_argument("--lambda-max", type=float, default=4.0)
    parser.add_argument("--k-aug", type=int, default=15)
    parser.add_argument("--sigma", choices=["abs", "sin"], default="abs",
                        help="abs: 0.2 + 0.6|x|; sin: 0.2 + 0.6|sin(2x)|.")
    # AA-CRC
    parser.add_argument("--aacrc-basis", choices=["poly", "rf", "both"], default="both")
    parser.add_argument("--aacrc-poly-split", choices=["cal", "union"], default="cal",
                        help="Split used to fit theta in the polynomial arm (C or D u C).")
    parser.add_argument("--poly-ridge-intercept", action="store_true",
                        help="Penalize the intercept in the polynomial arm (old behaviour).")
    parser.add_argument("--rf-trees", type=int, default=3)
    parser.add_argument("--rf-depth", type=int, default=4)
    parser.add_argument("--rf-min-leaf", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "heteroscedastic_script",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def mu_true(x):
    return np.sin(np.pi * x / 2)


def sigma_true(x):
    if SIGMA_SCENARIO == "sin":
        return 0.2 + 0.6 * np.abs(np.sin(2 * x))
    return 0.2 + 0.6 * np.abs(x)


def generate_data(n, rng):
    x = rng.uniform(-2, 2, size=n)
    mu = mu_true(x)
    sigma = sigma_true(x)
    y = mu + sigma * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y, "mu": mu, "sigma": sigma})


def fit_mean_model(data):
    model = LinearGAM(s(0, n_splines=20), lam=0.6)
    model.fit(data[["x"]].to_numpy(), data["y"].to_numpy())
    return model


def predict_mean(model, x):
    return model.predict(np.asarray(x).reshape(-1, 1))


def bounded_excess_loss(residual, lam):
    return np.minimum(1.0, np.maximum(0.0, residual - lam) / EXCESS_SCALE)


def oracle_unbounded_risk(lam, sigma, bias):
    z_plus = (lam - bias) / sigma
    z_minus = (lam + bias) / sigma
    return (
        sigma * (stats.norm.pdf(z_plus) + stats.norm.pdf(z_minus))
        - lam * (stats.norm.cdf(-z_plus) + stats.norm.cdf(-z_minus))
        + bias * (stats.norm.cdf(-z_plus) - stats.norm.cdf(-z_minus))
    )


def oracle_risk_matrix(x, lambda_grid, mean_model):
    sigma = sigma_true(x)
    bias = predict_mean(mean_model, x) - mu_true(x)
    risk = np.empty((len(x), len(lambda_grid)))
    for j, lam in enumerate(lambda_grid):
        risk[:, j] = (
            oracle_unbounded_risk(lam, sigma, bias)
            - oracle_unbounded_risk(lam + EXCESS_SCALE, sigma, bias)
        ) / EXCESS_SCALE
    return np.clip(risk, 0.0, B_LOSS)


def fit_tabicl_surface(context, mu_context, k_aug, lambda_max, device, rng):
    residual = np.abs(context["y"].to_numpy() - mu_context)
    x_rep = np.repeat(context["x"].to_numpy(), k_aug)
    residual_rep = np.repeat(residual, k_aug)
    lambda_rep = rng.uniform(0.0, lambda_max, size=len(x_rep))
    features = np.column_stack([x_rep, lambda_rep])
    target = bounded_excess_loss(residual_rep, lambda_rep)
    model = TabICLRegressor(device=device, kv_cache=True, random_state=42)
    model.fit(features, target)
    return model


def estimate_tabicl_risk(model, x, lambda_grid):
    risk = np.empty((len(x), len(lambda_grid)))
    for j, lam in enumerate(lambda_grid):
        features = np.column_stack([x, np.full(len(x), lam)])
        risk[:, j] = model.predict(features)
    risk = np.clip(risk, 0.0, B_LOSS)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for i in range(len(x)):
        risk[i] = np.clip(-iso.fit_transform(lambda_grid, -risk[i]), 0.0, B_LOSS)
    return risk


def crc_select(losses, grid, alpha):
    upper = (len(losses) * losses.mean(axis=0) + B_LOSS) / (len(losses) + 1)
    valid = np.flatnonzero(upper <= alpha)
    return float(grid[valid[0]]) if len(valid) else float(grid[-1])


def invert_risk_matrix(risk, lambda_grid, a):
    mask = risk <= a
    first = np.argmax(mask, axis=1)
    first[~mask.any(axis=1)] = len(lambda_grid) - 1
    return lambda_grid[first]


def calibrate_rectified(residual_cal, risk_cal, lambda_grid, a_grid, alpha):
    losses = np.empty((len(residual_cal), len(a_grid)))
    for j, a in enumerate(a_grid):
        lam = invert_risk_matrix(risk_cal, lambda_grid, a)
        losses[:, j] = bounded_excess_loss(residual_cal, lam)
    upper = (len(residual_cal) * losses.mean(axis=0) + B_LOSS) / (len(residual_cal) + 1)
    valid = np.flatnonzero(upper <= alpha)
    return float(a_grid[valid[-1]]) if len(valid) else float(a_grid[0])


# -----------------------------------------------------------------------------
# AA-CRC: shared pieces
# -----------------------------------------------------------------------------


def pseudo_labels(residual, lambda_max):
    """Bounded excess loss as the FNR over AACRC_N_PSEUDO positive pseudo-pixels.

    With u = Phi(x)^T theta and lambda = lambda_max - u, a pseudo-pixel is missed
    iff lambda_max - r + offset < u, i.e. offset < r - lambda.
    """
    offsets = (np.arange(AACRC_N_PSEUDO) + 0.5) * EXCESS_SCALE / AACRC_N_PSEUDO
    labels = [np.ones((AACRC_N_PSEUDO, 1), dtype=float) for _ in residual]
    scores = [(lambda_max - value + offsets)[:, None] for value in residual]
    return labels, scores


def validate_pseudo_reduction(lambda_max):
    residual = np.linspace(0.0, lambda_max + EXCESS_SCALE, 201)
    lam = np.linspace(0.0, lambda_max, 201)
    offsets = (np.arange(AACRC_N_PSEUDO) + 0.5) * EXCESS_SCALE / AACRC_N_PSEUDO
    delta = residual[:, None] - lam[None, :]
    approximate = (offsets[None, None, :] < delta[:, :, None]).mean(axis=2)
    exact = bounded_excess_loss(residual[:, None], lam[None, :])
    error = np.abs(approximate - exact)
    if error.max() > 1.0 / AACRC_N_PSEUDO + 1e-12:
        raise AssertionError("Falha na redução pseudo-label do AA-CRC.")
    return float(error.max()), float(error.mean())


def threshold_to_lambda(u, lambda_max, lambda_grid):
    return np.clip(lambda_max - np.maximum(0.0, u), lambda_grid[0], lambda_grid[-1])


def run_slsqp(objective, gradient, theta, args, bounds=None, name="AA-CRC"):
    attempts = []
    start = time.time()
    result, valid = None, False
    for maxiter in (AACRC_MAXITER, 3 * AACRC_MAXITER):
        result = minimize(
            objective,
            theta,
            method="SLSQP",
            args=args,
            jac=gradient,
            bounds=bounds,
            options={"maxiter": maxiter, "disp": False},
            tol=1e-8,
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


# -----------------------------------------------------------------------------
# AA-CRC with the polynomial basis
# -----------------------------------------------------------------------------


def aacrc_features(x):
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, np.abs(x), x**2])


def J_ridge_free_intercept(theta, labels, scores, phi, alpha, n, rho):
    return J(theta, labels, scores, phi, alpha, n) + rho * np.sum(theta[1:] ** 2)


def J_prime_ridge_free_intercept(theta, labels, scores, phi, alpha, n, rho):
    grad = np.asarray(J_prime(theta, labels, scores, phi, alpha, n), dtype=float).copy()
    grad[1:] += 2.0 * rho * theta[1:]
    return grad


def fit_aacrc_poly(x, residual, lambda_grid, alpha, ridge_intercept=False):
    features = aacrc_features(x)
    scaler = StandardScaler().fit(features)
    phi = scaler.transform(features)
    phi = np.column_stack([np.ones(len(phi)), phi]).astype(float)
    lambda_max = float(lambda_grid[-1])
    labels, scores = pseudo_labels(residual, lambda_max)
    losses = bounded_excess_loss(residual[:, None], lambda_grid[None, :])
    lambda_init = crc_select(losses, lambda_grid, alpha)
    theta = np.zeros(phi.shape[1])
    theta[0] = max(1e-6, lambda_max - lambda_init)
    n = len(residual)
    if ridge_intercept:
        objective, gradient = J, J_prime
        args = (labels, scores, phi, alpha, n, "ridge", AACRC_RIDGE)
    else:
        objective, gradient = J_ridge_free_intercept, J_prime_ridge_free_intercept
        args = (labels, scores, phi, alpha, n, AACRC_RIDGE)
    result, elapsed, attempts = run_slsqp(objective, gradient, theta, args, name="AA-CRC (poly)")
    fit_lambda = threshold_to_lambda(phi @ result.x, lambda_max, lambda_grid)
    return {
        "theta": result.x,
        "scaler": scaler,
        "lambda_max": lambda_max,
        "time": elapsed,
        "objective": float(result.fun),
        "attempts": attempts,
        "fit_risk": float(bounded_excess_loss(residual, fit_lambda).mean()),
        "n_params": int(phi.shape[1]),
    }


def predict_aacrc_poly(fit, x, lambda_grid):
    phi = fit["scaler"].transform(aacrc_features(x))
    phi = np.column_stack([np.ones(len(phi)), phi])
    return threshold_to_lambda(phi @ fit["theta"], fit["lambda_max"], lambda_grid)


# -----------------------------------------------------------------------------
# AA-CRC with random-forest leaf indicators (Blot et al., Algorithm 1)
# -----------------------------------------------------------------------------


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class LeafMap:
    """Phi(x) = one-hot indicators of the leaves reached by x in each tree."""

    def __init__(self, x_res, r_res, n_trees, max_depth, min_leaf, seed):
        X = np.asarray(x_res, dtype=float).reshape(-1, 1)
        self.rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=seed,
        ).fit(X, r_res)
        self.encoder = _one_hot_encoder().fit(self.rf.apply(X))
        self.n_trees = n_trees

    def __call__(self, x):
        X = np.asarray(x, dtype=float).reshape(-1, 1)
        return self.encoder.transform(self.rf.apply(X)).astype(float)

    @property
    def n_leaves(self):
        return int(sum(len(c) for c in self.encoder.categories_))


def fit_aacrc_rf(x, residual, lambda_grid, alpha, leaf_map):
    """theta on C, no ridge, box bounds -- as in the authors' tabular notebook.

    Leaf indicators of each tree sum to one, so no separate intercept is added.
    """
    phi = leaf_map(x)
    lambda_max = float(lambda_grid[-1])
    labels, scores = pseudo_labels(residual, lambda_max)
    losses = bounded_excess_loss(residual[:, None], lambda_grid[None, :])
    lambda_init = crc_select(losses, lambda_grid, alpha)
    theta = np.full(phi.shape[1], max(1e-6, lambda_max - lambda_init) / leaf_map.n_trees)
    n = len(residual)
    bounds = [(0.0, lambda_max)] * phi.shape[1]
    result, elapsed, attempts = run_slsqp(
        J, J_prime, theta, (labels, scores, phi, alpha, n, None, None),
        bounds=bounds, name="AA-CRC (RF leaves)",
    )
    fit_lambda = threshold_to_lambda(phi @ result.x, lambda_max, lambda_grid)
    fit_loss = bounded_excess_loss(residual, fit_lambda)
    # Multiaccuracy check on C: risk inside each leaf (group) with >= 20 points.
    counts = phi.sum(axis=0)
    leaf_risk = (phi * fit_loss[:, None]).sum(axis=0) / np.maximum(counts, 1.0)
    populated = counts >= 20
    return {
        "theta": result.x,
        "leaf_map": leaf_map,
        "lambda_max": lambda_max,
        "time": elapsed,
        "objective": float(result.fun),
        "attempts": attempts,
        "fit_risk": float(fit_loss.mean()),
        "n_params": int(phi.shape[1]),
        "fit_worst_leaf_risk": float(leaf_risk[populated].max()) if populated.any() else float("nan"),
        "rate_theta_at_bound": float(np.mean(
            np.isclose(result.x, 0.0) | np.isclose(result.x, lambda_max)
        )),
    }


def predict_aacrc_rf(fit, x, lambda_grid):
    u = fit["leaf_map"](x) @ fit["theta"]
    return threshold_to_lambda(u, fit["lambda_max"], lambda_grid)


# -----------------------------------------------------------------------------
# Evaluation and trial loop
# -----------------------------------------------------------------------------


def evaluate(loss, width, lam, sigma, alpha, method, lambda_grid, n_bins=5):
    edges = np.quantile(sigma, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    bins = np.searchsorted(edges[1:-1], sigma, side="right")
    bin_risk = np.array([loss[bins == group].mean() for group in range(n_bins)])
    row = {
        "method": method,
        "risk": loss.mean(),
        "coverage": (loss == 0).mean(),
        "worst_bin_risk": bin_risk.max(),
        "mean_bin_excess": np.maximum(bin_risk - alpha, 0.0).mean(),
        "mean_width": width.mean(),
        "lambda_sd": lam.std(ddof=1),
        "lambda_min_rate": np.isclose(lam, lambda_grid[0]).mean(),
        "lambda_max_rate": np.isclose(lam, lambda_grid[-1]).mean(),
    }
    return row, bin_risk


def run_trial(args, seed, device, diagnostic=None):
    rng = np.random.default_rng(seed)
    lambda_grid = np.linspace(0.0, args.lambda_max, args.lambda_points)
    a_grid = np.linspace(0.0, 1.0, args.a_points)
    train = generate_data(args.n_train, rng)
    context = generate_data(args.n_context, rng)
    cal = generate_data(args.n_cal, rng)
    test = generate_data(args.n_test, rng)
    mean_model = fit_mean_model(train)

    mu_context = predict_mean(mean_model, context["x"])
    mu_cal = predict_mean(mean_model, cal["x"])
    mu_test = predict_mean(mean_model, test["x"])
    residual_context = np.abs(context["y"].to_numpy() - mu_context)
    residual_cal = np.abs(cal["y"].to_numpy() - mu_cal)
    residual_test = np.abs(test["y"].to_numpy() - mu_test)
    x_context, x_cal, x_test = (d["x"].to_numpy() for d in (context, cal, test))

    union_residual = np.concatenate([residual_context, residual_cal])
    union_x = np.concatenate([x_context, x_cal])
    crc_losses = bounded_excess_loss(union_residual[:, None], lambda_grid[None, :])
    lambda_crc = crc_select(crc_losses, lambda_grid, args.alpha)

    lambdas = {"crc": np.full(args.n_test, lambda_crc)}
    diagnostics = {"lambda_crc": lambda_crc}

    if args.aacrc_basis in ("poly", "both"):
        if args.aacrc_poly_split == "union":
            px, pr = union_x, union_residual
        else:
            px, pr = x_cal, residual_cal
        fit_poly = fit_aacrc_poly(px, pr, lambda_grid, args.alpha,
                                  ridge_intercept=args.poly_ridge_intercept)
        lambdas["aacrc_poly"] = predict_aacrc_poly(fit_poly, x_test, lambda_grid)
        diagnostics.update({f"aacrc_poly_{k}": v for k, v in fit_poly.items()
                            if k not in ("theta", "scaler")})

    if args.aacrc_basis in ("rf", "both"):
        # RF on D (as D_res in Algorithm 1); theta on C only.
        leaf_map = LeafMap(x_context, residual_context, args.rf_trees,
                           args.rf_depth, args.rf_min_leaf, seed)
        fit_rf = fit_aacrc_rf(x_cal, residual_cal, lambda_grid, args.alpha, leaf_map)
        lambdas["aacrc_rf"] = predict_aacrc_rf(fit_rf, x_test, lambda_grid)
        diagnostics.update({f"aacrc_rf_{k}": v for k, v in fit_rf.items()
                            if k not in ("theta", "leaf_map")})
        diagnostics["aacrc_rf_n_leaves"] = leaf_map.n_leaves

    risk_oracle_cal = oracle_risk_matrix(x_cal, lambda_grid, mean_model)
    risk_oracle_test = oracle_risk_matrix(x_test, lambda_grid, mean_model)
    a_oracle = calibrate_rectified(
        residual_cal, risk_oracle_cal, lambda_grid, a_grid, args.alpha
    )
    lambda_oracle = invert_risk_matrix(risk_oracle_test, lambda_grid, a_oracle)

    tabicl = fit_tabicl_surface(
        context, mu_context, args.k_aug, args.lambda_max, device, rng
    )
    risk_recirc_cal = estimate_tabicl_risk(tabicl, x_cal, lambda_grid)
    risk_recirc_test = estimate_tabicl_risk(tabicl, x_test, lambda_grid)
    a_recirc = calibrate_rectified(
        residual_cal, risk_recirc_cal, lambda_grid, a_grid, args.alpha
    )
    lambda_recirc = invert_risk_matrix(risk_recirc_test, lambda_grid, a_recirc)

    lambdas["recirc"] = lambda_recirc
    lambdas["oracle"] = lambda_oracle
    diagnostics.update({"a_recirc": a_recirc, "a_oracle": a_oracle})

    if diagnostic is not None:
        edges = np.quantile(test["sigma"], np.linspace(0.0, 1.0, 6))
        edges[0] -= 1e-9
        bins = np.searchsorted(edges[1:-1], test["sigma"], side="right")
        losses_by_budget = np.column_stack([
            bounded_excess_loss(
                residual_test,
                invert_risk_matrix(risk_recirc_test, lambda_grid, a),
            )
            for a in a_grid
        ])
        diagnostic.add_trial(
            a_grid,
            losses_by_budget,
            bin_groups(bins, 5),
            seed=seed,
            a_hat=a_recirc,
        )

    rows = []
    bin_rows = []
    widths = {}
    for method in METHOD_ORDER:
        lam = lambdas[method]
        loss = bounded_excess_loss(residual_test, lam)
        width = 2.0 * lam
        row, bin_risk = evaluate(
            loss,
            width,
            lam,
            test["sigma"].to_numpy(),
            args.alpha,
            method,
            lambda_grid,
        )
        rows.append(row)
        widths[method] = {"x": x_test, "width": width}
        bin_rows.extend(
            {"method": method, "bin": group + 1, "risk": value}
            for group, value in enumerate(bin_risk)
        )
    return pd.DataFrame(rows), pd.DataFrame(bin_rows), widths, diagnostics


def summarize(data, metrics):
    summary = data.melt(
        id_vars="method", value_vars=metrics, var_name="metric", value_name="value"
    )
    summary = summary.groupby(["method", "metric"], as_index=False)["value"].agg(
        mean="mean", std="std", count="count"
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["se"]
    return summary


def paired_differences(results, metrics, reference="recirc"):
    """Paired differences (method - reference) across trials, with 95% CI."""
    rows = []
    wide = results.pivot(index="trial", columns="method")
    for metric in metrics:
        for method in METHOD_ORDER:
            if method == reference:
                continue
            d = (wide[(metric, method)] - wide[(metric, reference)]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            rows.append({
                "metric": metric,
                "comparison": f"{method} - {reference}",
                "mean_diff": d.mean(),
                "se": se,
                "ci95_low": d.mean() - 1.96 * se,
                "ci95_high": d.mean() + 1.96 * se,
            })
    return pd.DataFrame(rows)


def plot_results(results, bin_results, output_dir, alpha):
    metrics = [
        ("risk", "Marginal risk"),
        ("mean_bin_excess", "Mean bin excess"),
        ("worst_bin_risk", "Worst-bin risk"),
        ("mean_width", "Mean band width"),
    ]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for position, method in enumerate(METHOD_ORDER):
            values = results.loc[results["method"] == method, metric].to_numpy()
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(position + jitter, values, color=METHOD_COLORS[method], alpha=0.7)
            ax.hlines(values.mean(), position - 0.25, position + 0.25, color="black", lw=2)
        if metric in {"risk", "worst_bin_risk"}:
            ax.axhline(alpha, color="gray", ls="--")
        ax.set_title(title)
        ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    grouped = bin_results.groupby(["method", "bin"])["risk"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in METHOD_ORDER:
        data = grouped[grouped["method"] == method]
        x = data["bin"].to_numpy()
        mean = data["mean"].to_numpy()
        sem = data["sem"].to_numpy()
        ax.plot(x, mean, "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=METHOD_COLORS[method], alpha=0.15)
    ax.axhline(alpha, color="gray", ls="--")
    ax.set(xlabel="Difficulty bin", ylabel="Risk", xticks=range(1, 6))
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
    global SIGMA_SCENARIO, METHOD_ORDER
    args = parse_args()
    SIGMA_SCENARIO = args.sigma
    METHOD_ORDER = ["crc"]
    if args.aacrc_basis in ("poly", "both"):
        METHOD_ORDER.append("aacrc_poly")
    if args.aacrc_basis in ("rf", "both"):
        METHOD_ORDER.append("aacrc_rf")
    METHOD_ORDER += ["recirc", "oracle"]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas não está disponível.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_error, mean_error = validate_pseudo_reduction(args.lambda_max)
    print(f"AA-CRC pseudo-label: max_error={max_error:.6f}, mean_error={mean_error:.6f}")
    all_metrics = []
    all_bins = []
    diagnostics = []
    risk_diagnostic = RiskCalibration()
    start = time.time()
    for trial in range(args.trials):
        seed = args.seed + trial + 1
        trial_start = time.time()
        metrics, bins, _, diag = run_trial(
            args, seed, device, diagnostic=risk_diagnostic
        )
        metrics["trial"] = trial
        metrics["seed"] = seed
        bins["trial"] = trial
        bins["seed"] = seed
        diag.update({"trial": trial, "seed": seed})
        all_metrics.append(metrics)
        all_bins.append(bins)
        diagnostics.append(diag)
        compact = metrics[["method", "risk", "worst_bin_risk", "mean_width"]]
        print(f"\nTrial {trial + 1}/{args.trials} ({time.time() - trial_start:.1f}s)")
        print(compact.round(4).to_string(index=False))

    risk_diagnostic.save(args.output_dir, make_plot=not args.no_plots)
    results = pd.concat(all_metrics, ignore_index=True)
    bin_results = pd.concat(all_bins, ignore_index=True)
    diagnostics_df = pd.DataFrame(diagnostics)
    metric_names = [
        "risk",
        "coverage",
        "worst_bin_risk",
        "mean_bin_excess",
        "mean_width",
        "lambda_sd",
        "lambda_min_rate",
        "lambda_max_rate",
    ]
    summary = summarize(results, metric_names)
    paired = paired_differences(results, ["risk", "worst_bin_risk", "mean_bin_excess", "mean_width"])
    results.to_csv(args.output_dir / "per_replication_metrics.csv", index=False)
    bin_results.to_csv(args.output_dir / "per_replication_bin_risks.csv", index=False)
    diagnostics_df.to_csv(args.output_dir / "per_replication_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)

    metadata = {
        "experiment": "synthetic_heteroscedastic_regression",
        "sigma_scenario": args.sigma,
        "protocol": {
            "crc": "D u C",
            "aacrc_poly": f"theta on {'D u C' if args.aacrc_poly_split == 'union' else 'C'}; "
                          f"ridge={AACRC_RIDGE} {'with' if args.poly_ridge_intercept else 'without'} intercept",
            "aacrc_rf": "RF (Blot et al. Alg. 1) on |residuals| of D; theta on C; no ridge; bounds [0, lambda_max]",
            "recirc": "risk surface on D; budget on C",
        },
        "arguments": {**vars(args), "output_dir": str(args.output_dir), "device_resolved": device},
        "aacrc_commit": AACRC_COMMIT,
        "package_versions": {
            name: package_version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "pygam", "tabicl", "torch"]
        },
    }
    with (args.output_dir / "protocol_config.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    if not args.no_plots:
        plot_results(results, bin_results, args.output_dir, args.alpha)

    print("\nSummary")
    print(summary.round(4).to_string(index=False))
    print("\nPaired differences (method - recirc)")
    print(paired.round(4).to_string(index=False))
    print(f"\nTempo total: {time.time() - start:.1f}s")
    print(f"Resultados: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
