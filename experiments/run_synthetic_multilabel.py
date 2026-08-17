#!/usr/bin/env python3

"""Synthetic multilabel experiment runner.

This script generates synthetic multilabel data, runs CRC/AA-CRC/ReCIRC
benchmarks and saves results and plots.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
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

METHOD_ORDER = ["crc", "aacrc", "recirc", "oracle"]
METHOD_LABELS = {
    "crc": "Standard CRC",
    "aacrc": "AA-CRC",
    "recirc": "ReCIRC TabICL",
    "oracle": "Oracle Rectified",
}
METHOD_COLORS = {
    "crc": "#999999",
    "aacrc": "#009E73",
    "recirc": "#002F6C",
    "oracle": "#D55E00",
}
FEATURE_COLUMNS = [
    "x",
    "lambda",
    "score_mean",
    "score_std",
    "score_max",
    "score_top5_mean",
    "score_mass",
    "score_gap_top2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Experimento sintético multilabel.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dgp-seed", type=int, default=123)
    parser.add_argument("--n-context", type=int, default=500)
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "multilabel_script",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def generate_data(n, labels, rng, label_probs=None):
    x = rng.uniform(0.0, 1.0, size=n)
    mean_relevant = 2.0 + 6.0 * x
    n_relevant = np.clip(rng.poisson(mean_relevant), 1, labels)
    if label_probs is None:
        popularity = rng.normal(0.0, 0.7, size=labels)
        label_probs = expit(popularity)
        label_probs /= label_probs.sum()
    y = np.zeros((n, labels), dtype=int)
    for i in range(n):
        positive = rng.choice(labels, size=n_relevant[i], replace=False, p=label_probs)
        y[i, positive] = 1
    separation = 3.0 * (1.0 - x) + 0.6
    noise_sd = 0.5 + 1.3 * x
    logits = np.empty((n, labels))
    for i in range(n):
        logits[i] = separation[i] * (2 * y[i] - 1) + rng.normal(0.0, noise_sd[i], labels)
    return {
        "x": x,
        "difficulty": x,
        "y": y,
        "scores": expit(logits),
        "n_relevant": n_relevant,
        "label_probs": label_probs,
    }


def concatenate(*splits):
    return {
        "x": np.concatenate([data["x"] for data in splits]),
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


def build_augmented_context(context, lambda_grid, k_aug, rng):
    rows = []
    targets = []
    for i, x in enumerate(context["x"]):
        lambdas = rng.choice(lambda_grid, size=k_aug, replace=True)
        summary = score_summary(context["scores"][i])
        for lam in lambdas:
            loss = missed_positive_loss(
                context["scores"][i : i + 1], context["y"][i : i + 1], lam
            )[0]
            rows.append(np.concatenate([[x, lam], summary]))
            targets.append(loss)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), np.asarray(targets)


def risk_features(data, lambda_grid):
    rows = []
    indices = []
    for i, x in enumerate(data["x"]):
        summary = score_summary(data["scores"][i])
        for lam in lambda_grid:
            rows.append(np.concatenate([[x, lam], summary]))
            indices.append(i)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), np.asarray(indices)


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


def aacrc_features(data):
    rows = []
    for x, scores in zip(data["x"], data["scores"]):
        rows.append(np.concatenate([[x], score_summary(scores)]))
    return np.asarray(rows)


def fit_aacrc(data, alpha):
    features = aacrc_features(data)
    scaler = StandardScaler().fit(features)
    phi = scaler.transform(features)
    phi = np.column_stack([np.ones(len(phi)), phi]).astype(np.float64)
    labels = data["y"][:, :, None].astype(np.float64)
    scores = data["scores"][:, :, None].astype(np.float64)
    theta = np.zeros(phi.shape[1], dtype=np.float64)
    theta[0] = 0.5
    attempts = []
    start = time.time()
    result = None
    for maxiter in (AACRC_MAXITER, 3 * AACRC_MAXITER):
        result = minimize(
            J,
            theta,
            method="SLSQP",
            args=(labels, scores, phi, alpha, len(data["x"]), "ridge", AACRC_RIDGE),
            jac=J_prime,
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
        raise RuntimeError("AA-CRC falhou: " + " | ".join(attempts))
    return {
        "theta": result.x,
        "scaler": scaler,
        "time": time.time() - start,
        "objective": float(result.fun),
        "attempts": len(attempts),
    }


def predict_aacrc(fit, data, lambda_grid):
    phi = fit["scaler"].transform(aacrc_features(data))
    phi = np.column_stack([np.ones(len(phi)), phi])
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


def replicate_at_x(x, label_probs, n_mc, labels, rng):
    mean_relevant = 2.0 + 6.0 * x
    separation = 3.0 * (1.0 - x) + 0.6
    noise_sd = 0.5 + 1.3 * x
    n_relevant = np.clip(rng.poisson(mean_relevant, size=n_mc), 1, labels)
    y = np.zeros((n_mc, labels), dtype=int)
    for i in range(n_mc):
        positive = rng.choice(labels, size=n_relevant[i], replace=False, p=label_probs)
        y[i, positive] = 1
    logits = separation * (2 * y - 1) + rng.normal(0.0, noise_sd, size=(n_mc, labels))
    return y, expit(logits)


def oracle_risk_matrix(data, lambda_grid, label_probs, n_mc, labels, rng):
    risk = np.empty((len(data["x"]), len(lambda_grid)))
    for i, x in enumerate(data["x"]):
        y, scores = replicate_at_x(x, label_probs, n_mc, labels, rng)
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


def run_trial(args, seed, device, label_probs):
    rng = np.random.default_rng(seed)
    lambda_grid = np.linspace(0.0, 1.0, args.lambda_points)
    a_grid = np.linspace(0.0, 1.0, args.a_points)
    context = generate_data(args.n_context, args.labels, rng, label_probs)
    cal = generate_data(args.n_cal, args.labels, rng, label_probs)
    test = generate_data(args.n_test, args.labels, rng, label_probs)
    fit_data = concatenate(context, cal)

    crc_losses = losses_over_lambda(fit_data["scores"], fit_data["y"], lambda_grid)
    lambda_crc = crc_select(crc_losses, lambda_grid, args.alpha)
    lambdas = {"crc": np.full(args.n_test, lambda_crc)}

    aacrc_fit = fit_aacrc(fit_data, args.alpha)
    lambdas["aacrc"] = predict_aacrc(aacrc_fit, test, lambda_grid)

    tabicl = fit_tabicl_surface(context, lambda_grid, args.k_aug, device, seed, rng)
    recirc_risk_cal = estimate_tabicl_risk(tabicl, cal, lambda_grid)
    recirc_risk_test = estimate_tabicl_risk(tabicl, test, lambda_grid)
    a_recirc = calibrate_rectified(cal, recirc_risk_cal, lambda_grid, a_grid, args.alpha)
    lambdas["recirc"] = invert_risk_matrix(recirc_risk_test, lambda_grid, a_recirc)

    a_oracle = np.nan
    if not args.skip_oracle:
        oracle_rng = np.random.default_rng(seed + 9999)
        oracle_risk_cal = oracle_risk_matrix(
            cal, lambda_grid, label_probs, args.n_mc_oracle, args.labels, oracle_rng
        )
        oracle_risk_test = oracle_risk_matrix(
            test, lambda_grid, label_probs, args.n_mc_oracle, args.labels, oracle_rng
        )
        a_oracle = calibrate_rectified(cal, oracle_risk_cal, lambda_grid, a_grid, args.alpha)
        lambdas["oracle"] = invert_risk_matrix(oracle_risk_test, lambda_grid, a_oracle)

    rows = []
    bin_rows = []
    for method, lam in lambdas.items():
        row, bin_risk, bin_size = evaluate(
            test, lam, args.alpha, method, args.bins, lambda_grid
        )
        rows.append(row)
        for group, (risk, size) in enumerate(zip(bin_risk, bin_size), start=1):
            bin_rows.append({"method": method, "bin": group, "risk": risk, "set_size": size})
    diagnostics = {
        "lambda_crc": lambda_crc,
        "a_recirc": a_recirc,
        "a_oracle": a_oracle,
        "aacrc_time": aacrc_fit["time"],
        "aacrc_attempts": aacrc_fit["attempts"],
        "aacrc_objective": aacrc_fit["objective"],
    }
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
    args = parse_args()
    if args.labels < 5:
        raise ValueError("--labels deve ser pelo menos 5.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas não está disponível.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_rng = np.random.default_rng(args.dgp_seed)
    label_probs = generate_data(1, args.labels, label_rng)["label_probs"]
    validation = generate_data(
        128, args.labels, np.random.default_rng(args.dgp_seed + 1), label_probs
    )
    mapping_error = validate_aacrc_mapping(validation)
    print(f"AA-CRC loss mapping max error: {mapping_error:.3e}")

    all_metrics = []
    all_bins = []
    diagnostics = []
    start = time.time()
    for trial in range(args.trials):
        seed = args.seed + trial + 1
        trial_start = time.time()
        metrics, bins, diag = run_trial(args, seed, device, label_probs)
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

    results = pd.concat(all_metrics, ignore_index=True)
    bin_results = pd.concat(all_bins, ignore_index=True)
    diagnostics_df = pd.DataFrame(diagnostics)
    metric_names = ["risk", "worst_bin_risk", "mean_bin_excess", "mean_set_size"]
    summary = summarize(results, metric_names)
    paired_rows = [
        paired_difference(results, "recirc", "aacrc", metric)
        for metric in metric_names
    ]
    paired_rows.append(paired_difference(results, "aacrc", "crc", "worst_bin_risk"))
    paired = pd.DataFrame(paired_rows)

    results.to_csv(args.output_dir / "per_replication_metrics.csv", index=False)
    bin_results.to_csv(args.output_dir / "per_replication_bin_metrics.csv", index=False)
    diagnostics_df.to_csv(args.output_dir / "per_replication_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)

    metadata = {
        "experiment": "synthetic_multilabel_classification",
        "protocol": "fair_labeled_budget_D_union_C",
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
