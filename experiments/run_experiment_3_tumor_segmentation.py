#!/usr/bin/env python3
"""Experiment 3: risk-controlling tumor segmentation.

Dataset
-------
The downloaded ``polyps-pranet.npz`` contains 1,798 colorectal-polyp images in the
form used by the accompanying tumor-segmentation notebook.  ``sgmd`` stores a
PraNet sigmoid probability map for every image and ``targets`` stores its binary
ground-truth tumor mask.  Both arrays have shape ``(1798, 352, 352)``.  The
experiment consumes the cached maps, not the original RGB images, so it
evaluates uncertainty calibration rather than training PraNet itself.

Procedure
---------
Each seeded trial randomly divides the images into three disjoint parts:
700 context images, 700 calibration images, and the remaining 398 test images.
Every probability map is summarized by 90 quantiles, giving label-free image
features for the adaptive methods.

The script compares three methods at target false-negative risk ``alpha=0.10``:

* Standard CRC calibrates one global protective lambda on the calibration set.
* AA-CRC learns an image-dependent threshold from 90 probability quantiles;
  its objective is evaluated on nearest-neighbor-resized 64 x 64 masks.  This
  is a valid label-free AA-CRC feature map, but differs from the authors' polyp
  experiment, which used separately trained ResNet image embeddings.
* Rectified CRC (Route 2) fits a TabICL conditional-risk surface on randomly
  augmented context examples, then rectifies its image-specific lambdas using
  the independent calibration set.

Here a protective lambda defines ``C_lambda(x) = {pixel: p(pixel) >= 1-lambda}``.
Larger lambda therefore creates a larger mask and cannot increase the FNR.  FNR
is the fraction of true tumor pixels excluded from the predicted mask.

Outputs
-------
Per-trial marginal and conditional results, a four-metric summary, and first-
trial masks/lambdas are written below the configured output directory. The
headline metrics are marginal FNR, worst
difficulty-bin FNR, mean excess over alpha across bins, and mean predicted-mask
area.  Difficulty bins are fitted from a label-free uncertainty score on the
context probability maps.  Publication-style summary plots, local-behavior
plots, a lambda histogram, and any available image examples are saved in the
same experiment output directory.

By default, data are downloaded once to ``~/.cache/recirc`` and results are
written to ``results/experiment_3_tumor_segmentation``. Both locations can be
overridden from the command line, so no dataset needs to be committed to Git.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def ensure_packages():
    """Install missing runtime dependencies, matching the other experiment scripts."""
    packages = {
        "gdown": "gdown",
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "tabicl": "tabicl",
    }
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing missing package: {pip_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_packages()

try:
    from .risk_calibration import calibration_rows, bin_groups, save_risk_calibration as save_calibration
except ImportError:
    from risk_calibration import calibration_rows, bin_groups, save_risk_calibration as save_calibration

import gdown
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from skimage.transform import resize
from tabicl import TabICLRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "RECIRC_CACHE_DIR",
        Path.home() / ".cache" / "recirc",
    )
) / "experiment_3_tumor_segmentation"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_3_tumor_segmentation"
DATA_ARCHIVE_GOOGLE_DRIVE_ID = "1h7S6N_Rx7gdfO3ZunzErZy6H7620EbZK"
METHOD_ORDER = ["Standard CRC", "AA-CRC", "Rectified CRC"]


@dataclass(frozen=True)
class ExperimentConfig:
    alpha: float = 0.10
    loss_bound: float = 1.0
    n_trials: int = 20
    base_seed: int = 2026
    n_context: int = 700
    n_calibration: int = 700
    n_bins: int = 5
    n_quantiles: int = 90
    n_augmentations: int = 20
    aa_image_size: int = 64
    aa_regularization: str = "ridge"
    aa_regularization_strength: float = 0.01
    aa_max_iterations: int = 1000


def parse_args():
    """Parse the same run/output controls exposed by the other experiments."""
    parser = argparse.ArgumentParser(
        description="Experiment 3: tumor segmentation with CRC, AA-CRC, and ReCIRC."
    )
    parser.add_argument("--trials", type=int, default=ExperimentConfig.n_trials)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n-context", type=int, default=700)
    parser.add_argument("--n-cal", type=int, default=700)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--quantiles", type=int, default=90)
    parser.add_argument("--k-aug", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Use an existing polyps-pranet.npz instead of downloading the archive.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--keep-download", action="store_true",
                        help="Keep the 1.3 GB compressed archive after extraction.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Run only ReCIRC and save held-out risk-calibration diagnostics.")
    return parser.parse_args()


# Data preparation
def _safe_archive_members(archive):
    """Select only the cached NPZ and visualization JPEGs from the archive."""
    selected = []
    for member in archive.getmembers():
        normalized = Path(member.name)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"Unsafe path in downloaded archive: {member.name}")
        if normalized.name == "polyps-pranet.npz" or (
            normalized.parent.name == "examples"
            and normalized.parent.parent.name == "polyps"
            and normalized.suffix.lower() in {".jpg", ".jpeg"}
        ):
            selected.append(member)
    return selected


def prepare_tumor_data(data_dir, data_file=None, keep_download=False):
    """Return a local NPZ path, downloading and extracting the data if needed.

    The Google Drive artifact is the same archive used by the original tumor
    segmentation notebook. Only the probability/mask NPZ and optional example
    JPEGs are extracted; the archive is removed afterward by default to avoid
    retaining both a 1.3 GB download and its 3.4 GB extracted NPZ.
    """
    if data_file is not None:
        path = data_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--data-file not found: {path}")
        return path

    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "polyps-pranet.npz"
    if npz_path.is_file():
        print(f"Using cached tumor data: {npz_path}")
        return npz_path

    archive_path = data_dir / "data.tar.gz"
    if not archive_path.is_file():
        print("Downloading tumor-segmentation data (~1.3 GB)...")
        downloaded = gdown.download(
            id=DATA_ARCHIVE_GOOGLE_DRIVE_ID,
            output=str(archive_path),
            quiet=False,
        )
        if downloaded is None or not archive_path.is_file():
            raise RuntimeError("Tumor data download failed.")

    print(f"Extracting tumor data to: {data_dir}")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _safe_archive_members(archive)
        npz_members = [member for member in members if Path(member.name).name == "polyps-pranet.npz"]
        if len(npz_members) != 1:
            raise RuntimeError("Downloaded archive does not contain exactly one polyps-pranet.npz.")
        for member in members:
            source = archive.extractfile(member)
            if source is None:
                continue
            member_path = Path(member.name)
            if member_path.name == "polyps-pranet.npz":
                destination = npz_path
            else:
                destination = data_dir / "examples" / member_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as stream:
                shutil.copyfileobj(source, stream)

    if not keep_download:
        archive_path.unlink()
    if not npz_path.is_file():
        raise RuntimeError(f"Extraction completed without creating {npz_path}")
    return npz_path


def load_tumor_data(path):
    """Load PraNet probability maps and binary masks from the cached NPZ file."""
    if not path.exists():
        raise FileNotFoundError(f"Tumor data not found: {path}")
    with np.load(path) as data:
        if "sgmd" not in data or "targets" not in data:
            raise ValueError("Tumor NPZ must contain 'sgmd' and 'targets' arrays.")
        probabilities = data["sgmd"]
        masks = data["targets"].astype(bool)
    if probabilities.shape != masks.shape or probabilities.ndim != 3:
        raise ValueError(
            f"Incompatible tumor arrays: probabilities={probabilities.shape}, masks={masks.shape}"
        )
    return probabilities, masks


def compute_probability_quantiles(probability_maps, n_quantiles):
    """Represent each 352 x 352 probability map with label-free quantiles."""
    levels = np.linspace(0.0, 1.0, n_quantiles)
    values = np.empty((len(probability_maps), n_quantiles), dtype=float)
    for index, probability_map in enumerate(probability_maps):
        values[index] = np.quantile(probability_map.reshape(-1), levels)
    return pd.DataFrame(values, columns=[f"probability_quantile_{level:.3f}" for level in levels])


def make_three_way_split(n_samples, n_context, n_calibration, seed):
    """Return disjoint context, calibration, and test image indices."""
    if n_context + n_calibration >= n_samples:
        raise ValueError("Context and calibration sizes must leave at least one test image.")
    indices = np.random.default_rng(seed).permutation(n_samples)
    return (indices[:n_context], indices[n_context:n_context + n_calibration],
            indices[n_context + n_calibration:])


# Metrics
def false_negative_rate_from_masks(predicted_masks, true_masks):
    predicted_masks, true_masks = predicted_masks.astype(bool), true_masks.astype(bool)
    true_area = true_masks.reshape(len(true_masks), -1).sum(axis=1)
    missed_area = (true_masks & ~predicted_masks).reshape(len(true_masks), -1).sum(axis=1)
    return np.divide(missed_area, true_area, out=np.zeros(len(true_masks), dtype=float), where=true_area > 0)


def fit_uncertainty_edges(probability_maps, n_bins):
    uncertainty = (1.0 - 2.0 * np.abs(probability_maps.reshape(len(probability_maps), -1) - 0.5)).mean(axis=1)
    edges = np.quantile(uncertainty, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.maximum.accumulate(edges)


def apply_uncertainty_bins(probability_maps, edges):
    uncertainty = (1.0 - 2.0 * np.abs(probability_maps.reshape(len(probability_maps), -1) - 0.5)).mean(axis=1)
    return np.clip(np.searchsorted(edges[1:-1], uncertainty, side="right"), 0, len(edges) - 2)


def conditional_metrics(predicted_masks, true_masks, bins, method, alpha, lambdas):
    losses = false_negative_rate_from_masks(predicted_masks, true_masks)
    sizes = predicted_masks.reshape(len(predicted_masks), -1).sum(axis=1)
    true_sizes = true_masks.reshape(len(true_masks), -1).sum(axis=1)
    rows = []
    for bin_index in np.unique(bins):
        selected = bins == bin_index
        risk = float(losses[selected].mean())
        efficiency = np.divide(true_sizes[selected], sizes[selected], out=np.zeros(selected.sum()),
                               where=sizes[selected] > 0)
        rows.append({"method": method, "bin": int(bin_index), "conditional_risk": risk,
                     "mean_set_size": float(sizes[selected].mean()),
                     "region_area": float(sizes[selected].mean()), "pixelwise_coverage": 1.0 - risk,
                     "efficiency_y_over_c": float(efficiency.mean()),
                     "lambda_mean": float(np.asarray(lambdas)[selected].mean()),
                     "excess": max(risk - alpha, 0.0)})
    return pd.DataFrame(rows)


def summarize_four_metrics(marginal, conditional):
    local = (conditional.groupby(["trial", "method"], as_index=False)
             .agg(worst_local_risk=("conditional_risk", "max"), excess=("excess", "mean")))
    combined = marginal.merge(local, on=["trial", "method"], how="inner")
    return (combined.groupby("method", as_index=False)
            .agg(marginal_risk=("marginal_risk", "mean"),
                 worst_local_risk=("worst_local_risk", "mean"), excess=("excess", "mean"),
                 size=("size", "mean"), marginal_risk_sd=("marginal_risk", "std"),
                 worst_local_risk_sd=("worst_local_risk", "std"), excess_sd=("excess", "std"),
                 size_sd=("size", "std")))


# Shared segmentation operations and Standard CRC
def segmentation_mask(probability_maps, protective_lambda):
    protective_lambda = np.asarray(protective_lambda)
    if protective_lambda.ndim == 1:
        protective_lambda = protective_lambda[:, None, None]
    return probability_maps >= 1.0 - protective_lambda


def false_negative_rate(probability_maps, true_masks, protective_lambda):
    return false_negative_rate_from_masks(segmentation_mask(probability_maps, protective_lambda), true_masks)


def losses_over_lambda_segmentation(probability_maps, true_masks, lambda_grid):
    return np.column_stack([false_negative_rate(probability_maps, true_masks, value) for value in lambda_grid])


def calibrate_crc(calibration_losses, lambda_grid, alpha, loss_bound=1.0):
    corrected = (len(calibration_losses) * calibration_losses.mean(axis=0) + loss_bound) / (len(calibration_losses) + 1)
    valid = corrected <= alpha
    return float(lambda_grid[np.argmax(valid)]) if valid.any() else float(lambda_grid[-1])


# AA-CRC
def resize_aa_crc_training_maps(probability_maps, true_masks, image_size):
    probabilities = np.empty((len(probability_maps), image_size, image_size), dtype=float)
    masks = np.empty((len(true_masks), image_size, image_size), dtype=bool)
    for index in range(len(probability_maps)):
        probabilities[index] = resize(probability_maps[index], (image_size, image_size), order=1,
                                      anti_aliasing=True, preserve_range=True)
        masks[index] = resize(true_masks[index].astype(float), (image_size, image_size), order=0,
                              anti_aliasing=False, preserve_range=True) > 0.5
    return probabilities, masks


def prepare_aa_crc_features(training_features, target_features):
    training = np.asarray(training_features, dtype=float)
    center, scale = training.mean(axis=0), training.std(axis=0)
    scale[scale == 0.0] = 1.0

    def transform(features):
        values = (np.asarray(features, dtype=float) - center) / scale
        return np.concatenate([np.ones((len(values), 1)), values], axis=1)

    return transform(training), transform(target_features), center, scale


def aa_crc_objective(theta, masks, probabilities, embedding, alpha, regularization, strength):
    thresholds = np.maximum(embedding @ theta, -100.0)
    target = alpha - 1.0 / len(masks)
    values = []
    for mask, probability_map, threshold in zip(masks, probabilities, thresholds):
        if threshold <= 0.0:
            values.append(-(100.0 + threshold) * target)
            continue
        true_probabilities = probability_map[mask]
        integral = threshold - np.minimum(true_probabilities, threshold).mean()
        values.append(integral - threshold * target - 100.0 * target)
    penalty = strength * np.linalg.norm(theta) ** 2 if regularization == "ridge" else 0.0
    return float(np.mean(values) + penalty)


def aa_crc_gradient(theta, masks, probabilities, embedding, alpha, regularization, strength):
    thresholds = np.maximum(embedding @ theta, 0.0)
    losses = false_negative_rate_from_masks(probabilities >= thresholds[:, None, None], masks)
    gradient = np.mean(embedding * (losses - (alpha - 1.0 / len(masks)))[:, None], axis=0)
    return gradient + (2 * strength * theta if regularization == "ridge" else 0.0)


def fit_aa_crc_segmentation(probability_maps_train, true_masks_train, features_train,
                            probability_maps_target, features_target, config, rng):
    """Fit AA-CRC using probability quantiles as its adaptive feature map.

    AA-CRC permits a chosen label-free representation of each input.  Here the
    representation is 90 quantiles of the PraNet probability map rather than
    the pretrained ResNet embeddings used in the authors' original polyp
    notebook.  The AA-CRC objective and image-specific threshold mechanism are
    unchanged, but results should be described as quantile-feature AA-CRC.
    """
    train_embedding, target_embedding, _, _ = prepare_aa_crc_features(features_train, features_target)
    initial_theta = rng.uniform(0.0, 1.0, size=train_embedding.shape[1])
    result = minimize(aa_crc_objective, initial_theta, method="SLSQP",
                      args=(true_masks_train, probability_maps_train, train_embedding, config.alpha,
                            config.aa_regularization, config.aa_regularization_strength),
                      jac=aa_crc_gradient, options={"disp": False, "maxiter": config.aa_max_iterations},
                      tol=1e-10)
    threshold_target = np.clip(target_embedding @ result.x, 0.0, 1.0)
    lambdas = 1.0 - threshold_target
    return segmentation_mask(probability_maps_target, lambdas), lambdas


# Rectified CRC (Route 2)
def build_augmented(probability_maps, true_masks, features, n_augmentations, rng):
    """Create the notebook's random ``(image features, lambda) -> FNR`` data.

    Each context image receives ``n_augmentations`` independent protective
    lambdas sampled uniformly from [0, 1].  Repeating the image's 90 quantiles
    beside those lambdas gives TabICL examples whose targets are the observed
    per-image false-negative rates.
    """
    lambdas = rng.uniform(0.0, 1.0, (len(probability_maps), n_augmentations))
    repeated_features = np.repeat(features.to_numpy(), n_augmentations, axis=0)
    losses = np.empty(lambdas.size)
    for column in range(n_augmentations):
        losses[column::n_augmentations] = false_negative_rate(
            probability_maps, true_masks, lambdas[:, column])
    augmented = pd.DataFrame(repeated_features, columns=features.columns)
    augmented["lambda"] = lambdas.reshape(-1)
    return augmented, pd.Series(losses, name="false_negative_rate")


def fit_risk_model(probability_maps, true_masks, features, n_augmentations,
                   device, rng):
    """Fit TabICL to the randomly augmented context data."""
    augmented, losses = build_augmented(
        probability_maps, true_masks, features, n_augmentations, rng)
    model = TabICLRegressor(device=device, kv_cache=True, random_state=42)
    model.fit(augmented, losses)
    return model


def predict_risk_matrix(model, features, lambda_grid, loss_bound=1.0,
                        enforce_monotone=True):
    """Estimate each image's FNR curve over the protective-lambda grid.

    True FNR is nonincreasing because a larger protective lambda produces a
    larger mask.  The optional isotonic step projects each noisy TabICL curve
    onto that required shape before it is inverted.
    """
    features = features.reset_index(drop=True).copy()
    estimates = np.empty((len(features), len(lambda_grid)))
    for column, protective_lambda in enumerate(lambda_grid):
        query = features.copy()
        query["lambda"] = protective_lambda
        estimates[:, column] = model.predict(query)
    estimates = np.clip(estimates, 0.0, loss_bound)
    if enforce_monotone:
        for row in range(len(estimates)):
            estimates[row] = -IsotonicRegression(
                increasing=True, out_of_bounds="clip"
            ).fit_transform(lambda_grid, -estimates[row])
    return np.clip(estimates, 0.0, loss_bound)


def invert_risk_curve(risk, lambda_grid, risk_budget):
    """Select the smallest protective lambda with estimated FNR <= budget."""
    selected = []
    for row in risk:
        valid = row <= risk_budget
        selected.append(lambda_grid[np.argmax(valid)] if valid.any() else lambda_grid[-1])
    return np.asarray(selected, dtype=float)


def run_recirc(context_p, context_y, context_x, calibration_p, calibration_y,
               calibration_x, test_p, test_x, config, lambda_grid, a_grid,
               device, seed):
    """Fit Route 2, rectify its risk budget, and predict test masks.

    TabICL is trained only on the context split.  For every candidate budget
    ``a``, its calibration risk curves are inverted into image-specific lambdas
    and evaluated against calibration masks.  The largest budget satisfying the
    finite-sample CRC correction is then used for the independent test images.
    """
    model = fit_risk_model(
        context_p, context_y, context_x, config.n_augmentations, device,
        np.random.default_rng(seed))
    calibration_risk = predict_risk_matrix(
        model, calibration_x, lambda_grid, config.loss_bound)

    calibration_losses = np.empty((len(calibration_p), len(a_grid)))
    for column, risk_budget in enumerate(a_grid):
        selected = invert_risk_curve(calibration_risk, lambda_grid, risk_budget)
        calibration_losses[:, column] = false_negative_rate(
            calibration_p, calibration_y, selected)

    corrected = ((len(calibration_losses) * calibration_losses.mean(axis=0)
                  + config.loss_bound) / (len(calibration_losses) + 1))
    valid = corrected <= config.alpha
    a_hat = float(a_grid[np.where(valid)[0].max()]) if valid.any() else float(a_grid[0])

    test_risk = predict_risk_matrix(model, test_x, lambda_grid, config.loss_bound)
    lambdas = invert_risk_curve(test_risk, lambda_grid, a_hat)
    return segmentation_mask(test_p, lambdas), lambdas, test_risk, a_hat


def risk_calibration_diagnostic(test_p, test_y, test_risk, test_bins,
                                lambda_grid, a_grid, n_bins, a_hat):
    """Average per-image test FNR at every budget, with fixed context-fit bins."""
    # Evaluate masks only once per lambda; each budget then uses table lookups.
    loss_table = losses_over_lambda_segmentation(test_p, test_y, lambda_grid)
    losses = np.column_stack([
        loss_table[np.arange(len(test_p)), np.searchsorted(
            lambda_grid, invert_risk_curve(test_risk, lambda_grid, a))]
        for a in a_grid])
    return calibration_rows(a_grid, losses, bin_groups(test_bins, n_bins), a_hat), loss_table


def save_risk_calibration(curves, output_dir, make_plot=True):
    """Compatibility wrapper for existing experiment-3 callers."""
    save_calibration(curves, output_dir, make_plot, ylabel="Average per-image FNR")


def run_trial(probabilities, masks, features, config, seed, device):
    context_index, calibration_index, test_index = make_three_way_split(
        len(probabilities), config.n_context, config.n_calibration, seed)
    context_p, calibration_p, test_p = probabilities[context_index], probabilities[calibration_index], probabilities[test_index]
    context_y, calibration_y, test_y = masks[context_index], masks[calibration_index], masks[test_index]
    context_x, calibration_x, test_x = (features.iloc[index].reset_index(drop=True)
                                        for index in (context_index, calibration_index, test_index))
    lambda_grid, a_grid = np.linspace(0, 1, 51), np.linspace(0.001, 1, 100)
    test_bins = apply_uncertainty_bins(test_p, fit_uncertainty_edges(context_p, config.n_bins))
    predictions, lambdas = {}, {}

    standard_lambda = calibrate_crc(
        losses_over_lambda_segmentation(calibration_p, calibration_y, lambda_grid),
        lambda_grid, config.alpha, config.loss_bound)
    predictions["Standard CRC"] = segmentation_mask(test_p, standard_lambda)
    lambdas["Standard CRC"] = np.full(len(test_p), standard_lambda)

    aa_index = np.r_[context_index, calibration_index]
    aa_p, aa_y = resize_aa_crc_training_maps(probabilities[aa_index], masks[aa_index], config.aa_image_size)
    nonempty = aa_y.reshape(len(aa_y), -1).sum(axis=1) > 0
    predictions["AA-CRC"], lambdas["AA-CRC"] = fit_aa_crc_segmentation(
        aa_p[nonempty], aa_y[nonempty],
        features.iloc[aa_index].reset_index(drop=True).loc[nonempty].reset_index(drop=True),
        test_p, test_x, config, np.random.default_rng(seed))

    predictions["Rectified CRC"], lambdas["Rectified CRC"], test_risk, a_hat = run_recirc(
        context_p, context_y, context_x, calibration_p, calibration_y, calibration_x,
        test_p, test_x, config, lambda_grid, a_grid, device, seed)

    marginal, conditional = [], []
    for method in METHOD_ORDER:
        losses = false_negative_rate(test_p, test_y, lambdas[method])
        marginal.append({"method": method, "marginal_risk": float(losses.mean()),
                         "size": float(predictions[method].reshape(len(test_p), -1).sum(axis=1).mean())})
        conditional.append(conditional_metrics(
            predictions[method], test_y, test_bins, method, config.alpha, lambdas[method]))
    curves, loss_table = risk_calibration_diagnostic(
        test_p, test_y, test_risk, test_bins, lambda_grid, a_grid, config.n_bins, a_hat)
    diagnostics = {"risk_calibration": curves.to_records(index=False, column_dtypes={"group": "U32"}),
                   "test_risk": test_risk, "test_loss_table": loss_table,
                   "test_bins": test_bins, "lambda_grid": lambda_grid,
                   "a_grid": a_grid, "a_hat": a_hat,
                   "test_indices": test_index, "standard_lambda": standard_lambda,
                   "standard_masks": predictions["Standard CRC"],
                   "rectified_masks": predictions["Rectified CRC"],
                   "aa_crc_masks": predictions["AA-CRC"],
                   "rectified_lambdas": lambdas["Rectified CRC"],
                   "aa_crc_lambdas": lambdas["AA-CRC"]}
    return pd.DataFrame(marginal), pd.concat(conditional, ignore_index=True), diagnostics


# Figures
def plot_four_metrics(per_trial_metrics, alpha):
    """Plot the four headline metrics as jittered per-trial comparisons."""
    methods = [method for method in METHOD_ORDER
               if method in per_trial_metrics["method"].unique()]
    colors = {
        "Standard CRC": "#1f77b4",
        "AA-CRC": "#ff7f0e",
        "Rectified CRC": "#d62728",
    }
    panels = [
        ("test_fnr", "FNR", "(a) Marginal risk", True),
        ("worst_bin_risk", "FNR", "(b) Worst-bin risk", True),
        ("mean_excess_by_bin", "Mean excess", "(c) Mean excess by bin", False),
        ("avg_size", "Set size (pixels)", "(d) Mean mask size", False),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    jitter = np.random.default_rng(0)
    for axis, (metric, ylabel, title, show_alpha) in zip(axes.ravel(), panels):
        for position, method in enumerate(methods):
            values = per_trial_metrics.loc[
                per_trial_metrics["method"] == method, metric].to_numpy()
            x = np.full(len(values), position, dtype=float) + jitter.normal(0, 0.04, len(values))
            axis.scatter(x, values, alpha=0.75, s=45, color=colors[method])
            axis.hlines(values.mean(), position - 0.2, position + 0.2,
                        linewidth=2.5, color="black")
        if show_alpha:
            axis.axhline(alpha, linestyle="--", color="gray", linewidth=1.5)
        if metric == "mean_excess_by_bin":
            axis.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
        axis.set_xticks(range(len(methods)))
        axis.set_xticklabels(methods, fontsize=9, rotation=15)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.35)
    figure.suptitle(
        f"Tumor segmentation | alpha={alpha} | "
        f"{per_trial_metrics['trial'].nunique()} splits",
        fontsize=13, fontweight="bold", y=1.02)
    figure.tight_layout()
    return figure


def plot_local_behavior(conditional, alpha):
    """Plot coverage, area, efficiency, and lambda across difficulty bins."""
    labels = {value: str(value + 1) for value in sorted(conditional["bin"].unique())}
    labels[min(labels)] = "1 easiest"
    labels[max(labels)] = f"{max(labels) + 1} hardest"
    data = conditional.assign(difficulty_bin=conditional["bin"].map(labels))
    columns = ["pixelwise_coverage", "region_area", "efficiency_y_over_c", "lambda_mean"]
    summary = data.groupby(
        ["method", "bin", "difficulty_bin"], as_index=False)[columns].mean()
    specifications = [
        ("pixelwise_coverage", "Pixel-wise coverage"),
        ("region_area", "Average mask area (pixels)"),
        ("efficiency_y_over_c", "Efficiency |Y| / |C|"),
        ("lambda_mean", "Average protective lambda"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (metric, title) in zip(axes.ravel(), specifications):
        for method in METHOD_ORDER:
            values = summary[summary["method"] == method].sort_values("bin")
            axis.plot(values["difficulty_bin"].astype(str), values[metric],
                      marker="o", linewidth=2, label=method)
        if metric == "pixelwise_coverage":
            axis.axhline(1.0 - alpha, color="black", linestyle=":",
                         linewidth=1.5, label="target")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    axes[0, 0].legend()
    figure.suptitle(
        "Local behavior across probability-only difficulty bins | "
        f"mean over {conditional['trial'].nunique()} runs",
        y=1.02)
    figure.tight_layout()
    return figure


def plot_lambda_histogram(diagnostics):
    """Compare first-trial global and image-adaptive protective lambdas."""
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(diagnostics["rectified_lambdas"], bins=25, edgecolor="black",
              alpha=0.55, label="Rectified CRC")
    axis.hist(diagnostics["aa_crc_lambdas"], bins=25, edgecolor="black",
              alpha=0.45, label="AA-CRC")
    standard_lambda = float(diagnostics["standard_lambda"])
    axis.axvline(standard_lambda, color="black", linestyle=":", linewidth=2,
                 label=f"Standard CRC = {standard_lambda:.3f}")
    axis.set_xlabel("Protective lambda")
    axis.set_ylabel("Number of test images")
    axis.set_title("First-trial test lambdas by method")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_test_examples(diagnostics, probabilities, example_directory,
                       output_directory, n_examples=10):
    """Save notebook-style comparisons for test images with cached JPEGs.

    Each figure shows the RGB input, PraNet probability heatmap, masks from all
    three methods, and ground truth.  If the optional example JPEG directory is
    absent, the function returns without failing the experiment.
    """
    if not example_directory.exists():
        print(f"Skipping example figures; directory not found: {example_directory}")
        return
    available = [int(index) for index in diagnostics["test_indices"]
                 if (example_directory / f"{int(index)}.jpg").exists()
                 and (example_directory / f"{int(index)}_gt_mask.jpg").exists()]
    rng = np.random.default_rng(4)
    for number in range(min(n_examples, len(available))):
        image_index = int(rng.choice(available))
        position = int(np.where(diagnostics["test_indices"] == image_index)[0][0])
        image = imread(example_directory / f"{image_index}.jpg")
        ground_truth = imread(example_directory / f"{image_index}_gt_mask.jpg")
        probability = resize(probabilities[image_index], image.shape[:2], order=1,
                             anti_aliasing=True, preserve_range=True)
        masks = [
            resize(diagnostics[key][position].astype(float), image.shape[:2], order=0,
                   anti_aliasing=False, preserve_range=True) > 0.5
            for key in ("standard_masks", "rectified_masks", "aa_crc_masks")
        ]
        figure, axes = plt.subplots(1, 6, figsize=(16.8, 4.76))
        axes[0].imshow(image)
        heatmap = axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1)
        for axis, mask in zip(axes[2:5], masks):
            axis.imshow(mask, cmap="gray")
        axes[5].imshow(ground_truth, cmap="gray")
        titles = ["Input", "PraNet probability", "Standard CRC",
                  "Rectified CRC", "AA-CRC", "Ground truth"]
        for axis, title in zip(axes, titles):
            axis.set_title(title)
            axis.axis("off")
        figure.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        figure.suptitle(f"Test image {image_index}")
        figure.tight_layout()
        figure.savefig(output_directory / f"test_example_{number + 1}.png",
                       dpi=300, bbox_inches="tight")
        plt.close(figure)


def make_figures(per_trial_metrics, conditional, diagnostics, probabilities,
                 example_directory, config, results_directory):
    """Generate and save every experiment figure after the trials finish."""
    output_directory = results_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    four_metrics = plot_four_metrics(per_trial_metrics, config.alpha)
    four_metrics.savefig(output_directory / "four_metrics.pdf", bbox_inches="tight")
    four_metrics.savefig(output_directory / "four_metrics.png", dpi=300,
                         bbox_inches="tight")
    plt.close(four_metrics)

    local_behavior = plot_local_behavior(conditional, config.alpha)
    local_behavior.savefig(output_directory / "local_behavior_probability_bins.png",
                            dpi=300, bbox_inches="tight")
    plt.close(local_behavior)

    lambda_histogram = plot_lambda_histogram(diagnostics)
    lambda_histogram.savefig(output_directory / "test_lambda_histogram.png",
                             dpi=300, bbox_inches="tight")
    plt.close(lambda_histogram)

    plot_test_examples(diagnostics, probabilities, example_directory, output_directory)
    print(f"Figures saved to: {output_directory}")


def paired_differences(per_trial_metrics):
    """Summarize paired ReCIRC-minus-AA-CRC differences across identical splits."""
    rows = []
    for metric in ["test_fnr", "worst_bin_risk", "mean_excess_by_bin", "avg_size"]:
        paired = per_trial_metrics.pivot(index="trial", columns="method", values=metric).dropna()
        difference = paired["Rectified CRC"] - paired["AA-CRC"]
        standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
        rows.append({
            "comparison": "Rectified CRC - AA-CRC",
            "metric": metric,
            "n_pairs": len(difference),
            "mean_diff": float(difference.mean()),
            "sd_diff": float(difference.std(ddof=1)),
            "ci95_low": float(difference.mean() - 1.96 * standard_error),
            "ci95_high": float(difference.mean() + 1.96 * standard_error),
            "rectified_better_fraction": float((difference < 0.0).mean()),
        })
    return pd.DataFrame(rows)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-found"


def main():
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    config = ExperimentConfig(
        alpha=args.alpha,
        n_trials=args.trials,
        base_seed=args.seed,
        n_context=args.n_context,
        n_calibration=args.n_cal,
        n_bins=args.bins,
        n_quantiles=args.quantiles,
        n_augmentations=args.k_aug,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = prepare_tumor_data(args.data_dir, args.data_file, args.keep_download)
    probabilities, masks = load_tumor_data(data_path)
    features = compute_probability_quantiles(probabilities, config.n_quantiles)
    print(f"Tumor data: {probabilities.shape} | device: {device}")

    if args.diagnostic_only:
        curves_runs = []
        for trial in range(config.n_trials):
            seed = config.base_seed + trial
            d, c, t = make_three_way_split(len(probabilities), config.n_context,
                                          config.n_calibration, seed)
            lambda_grid, a_grid = np.linspace(0, 1, 51), np.linspace(0.001, 1, 100)
            edges = fit_uncertainty_edges(probabilities[d], config.n_bins)
            bins = apply_uncertainty_bins(probabilities[t], edges)
            _, _, risk, a_hat = run_recirc(
                probabilities[d], masks[d], features.iloc[d],
                probabilities[c], masks[c], features.iloc[c],
                probabilities[t], features.iloc[t], config, lambda_grid, a_grid, device, seed)
            curves, losses = risk_calibration_diagnostic(
                probabilities[t], masks[t], risk, bins, lambda_grid, a_grid, config.n_bins, a_hat)
            curves["trial"], curves["seed"] = trial, seed
            curves_runs.append(curves)
            np.savez_compressed(output_dir / f"risk_calibration_trial_{trial}.npz",
                                test_indices=t, test_risk=risk, test_loss_table=losses,
                                test_bins=bins, bin_edges=edges, lambda_grid=lambda_grid,
                                a_grid=a_grid, a_hat=a_hat, seed=seed)
            save_risk_calibration(pd.concat(curves_runs, ignore_index=True),
                                  output_dir, not args.no_plots)
            print(f"Diagnostic trial {trial + 1}/{config.n_trials} saved", flush=True)
        (output_dir / "risk_calibration_config.json").write_text(json.dumps({
            "config": vars(config), "device": device, "data_path": str(data_path),
            "mode": "diagnostic-only", "grouping": "context-fitted probability uncertainty bins",
        }, indent=2))
        return

    marginal_runs, conditional_runs, diagnostic_rows, first_diagnostics = [], [], [], None
    calibration_runs = []
    start = time.time()
    for trial in range(config.n_trials):
        trial_start = time.time()
        seed = config.base_seed + trial
        marginal, conditional, diagnostics = run_trial(
            probabilities, masks, features, config, seed, device)
        marginal["trial"], marginal["seed"] = trial, seed
        conditional["trial"], conditional["seed"] = trial, seed
        marginal_runs.append(marginal)
        conditional_runs.append(conditional)
        curves = pd.DataFrame.from_records(diagnostics["risk_calibration"])
        curves["trial"], curves["seed"] = trial, seed
        calibration_runs.append(curves)
        save_risk_calibration(pd.concat(calibration_runs, ignore_index=True),
                              output_dir, not args.no_plots)
        diagnostic_rows.append({
            "trial": trial,
            "seed": seed,
            "standard_lambda": float(diagnostics["standard_lambda"]),
            "aa_crc_lambda_mean": float(diagnostics["aa_crc_lambdas"].mean()),
            "aa_crc_lambda_sd": float(diagnostics["aa_crc_lambdas"].std()),
            "rectified_lambda_mean": float(diagnostics["rectified_lambdas"].mean()),
            "rectified_lambda_sd": float(diagnostics["rectified_lambdas"].std()),
        })
        if first_diagnostics is None:
            first_diagnostics = diagnostics

        pd.concat(marginal_runs, ignore_index=True).to_csv(
            output_dir / "per_replication_marginal_incremental.csv", index=False)
        pd.concat(conditional_runs, ignore_index=True).to_csv(
            output_dir / "per_replication_bin_metrics_incremental.csv", index=False)
        print(f"Trial {trial + 1}/{config.n_trials} ({time.time() - trial_start:.1f}s)")

    marginal = pd.concat(marginal_runs, ignore_index=True)
    conditional = pd.concat(conditional_runs, ignore_index=True)
    summary = summarize_four_metrics(marginal, conditional)
    per_trial_conditional = (conditional.groupby(["trial", "method"], as_index=False)
                             .agg(worst_bin_risk=("conditional_risk", "max"),
                                  mean_excess_by_bin=("excess", "mean")))
    final_metrics = (marginal.rename(columns={"marginal_risk": "test_fnr", "size": "avg_size"})
                     [["trial", "seed", "method", "test_fnr", "avg_size"]]
                     .merge(per_trial_conditional, on=["trial", "method"], how="left"))
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    paired = paired_differences(final_metrics)

    final_metrics.to_csv(output_dir / "per_replication_metrics.csv", index=False)
    conditional.to_csv(output_dir / "per_replication_bin_metrics.csv", index=False)
    diagnostics_frame.to_csv(output_dir / "per_replication_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_differences.csv", index=False)
    np.savez_compressed(output_dir / "first_trial_diagnostics.npz", **first_diagnostics)

    metadata = {
        "experiment": "tumor_segmentation",
        "dataset": {
            "source": "Google Drive archive from the original tumor-segmentation notebook",
            "google_drive_id": DATA_ARCHIVE_GOOGLE_DRIVE_ID,
            "local_npz": str(data_path),
            "shape": list(probabilities.shape),
        },
        "arguments": {
            **vars(args),
            "data_dir": str(args.data_dir),
            "data_file": str(args.data_file) if args.data_file else None,
            "output_dir": str(output_dir),
            "device_resolved": device,
        },
        "aa_crc_note": (
            "Valid AA-CRC with 90 label-free probability-quantile features; "
            "not the authors' separately trained ResNet embedding configuration."
        ),
        "package_versions": {
            name: package_version(name)
            for name in ["gdown", "numpy", "pandas", "scipy", "scikit-image", "scikit-learn", "tabicl", "torch"]
        },
        "elapsed_seconds": time.time() - start,
    }
    with (output_dir / "protocol_config.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print(summary.round(4).to_string(index=False))
    print("\nPaired differences")
    print(paired.round(4).to_string(index=False))
    if not args.no_plots:
        example_directory = data_path.parent / "examples"
        rgb_report = PROJECT_ROOT / "data/polyps/rgb_verification.json"
        if rgb_report.is_file():
            report = json.loads(rgb_report.read_text())
            if (report.get("complete") and report.get("targets_sha256") ==
                    hashlib.sha256(masks.tobytes()).hexdigest()):
                example_directory = rgb_report.parent / "examples"
        make_figures(final_metrics, conditional, first_diagnostics, probabilities,
                     example_directory, config, output_dir)
    print(f"\nTotal time: {time.time() - start:.1f}s")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
