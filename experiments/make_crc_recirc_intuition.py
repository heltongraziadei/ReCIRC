"""Create an intuitive, illustrative CRC-versus-ReCIRC example.

Part A is a pedagogical numerical example: real polyp images make the easy/hard
distinction concrete, while its displayed risks are hypothetical. Part B reads
the five-bin empirical results from Experiment 3 and summarizes them over trials.

Run from the repository root:

    MPLCONFIGDIR=.matplotlib-cache .venv/bin/python \
        experiments/make_crc_recirc_intuition.py

Outputs are written to ``results/crc_recirc_intuition``.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "polyps" / "examples"
OUTPUT_DIR = ROOT / "results" / "crc_recirc_intuition"
EXPERIMENT_3_BIN_RESULTS = (
    ROOT / "results" / "experiment_3_tumor_segmentation"
    / "per_replication_bin_metrics.csv"
)

ALPHA = 0.10
EASY_SHARE = 0.80
RISKS = {
    "Global CRC": {"easy": 0.01, "hard": 0.42},
    "ReCIRC": {"easy": 0.09, "hard": 0.11},
}
COLORS = {
    "easy": "#2E8B57",
    "hard": "#db0000",
    "crc": "#3667A6",
    "recirc": "#E68632",
    "miss": "#222222",
    "light": "#F4F6F8",
}


def _read_rgb(path):
    image = plt.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(float) / np.iinfo(image.dtype).max
    return image


def choose_image_examples():
    """Choose large- and small-lesion examples using ground-truth mask area."""
    candidates = []
    for mask_path in sorted(IMAGE_DIR.glob("*_gt_mask.jpg")):
        identifier = mask_path.name.removesuffix("_gt_mask.jpg")
        image_path = IMAGE_DIR / f"{identifier}.jpg"
        if not image_path.exists():
            continue
        mask = plt.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., :3].mean(axis=2)
        cutoff = 127 if np.issubdtype(mask.dtype, np.integer) else 0.5
        fraction = float(np.mean(mask > cutoff))
        if 0.001 < fraction < 0.65:
            candidates.append((fraction, image_path, mask_path))
    if len(candidates) < 2:
        raise FileNotFoundError(f"Could not find paired images and masks in {IMAGE_DIR}")
    candidates.sort(key=lambda item: item[0])
    # Avoid extreme masks that can be mostly annotation artifacts.
    hard = candidates[max(0, int(0.08 * (len(candidates) - 1)))]
    easy = candidates[min(len(candidates) - 1, int(0.90 * (len(candidates) - 1)))]
    return easy, hard


def show_overlay(axis, example, title, subtitle, color):
    fraction, image_path, mask_path = example
    image = _read_rgb(image_path)
    mask = plt.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., :3].mean(axis=2)
    cutoff = 127 if np.issubdtype(mask.dtype, np.integer) else 0.5
    mask = mask > cutoff
    axis.imshow(image)
    axis.contour(mask, levels=[0.5], colors=[color], linewidths=2.5)
    axis.set_title(title, color=color, fontsize=15, fontweight="bold", pad=8)
    axis.text(
        0.5, -0.07, f"{subtitle}\nlesion area: {100 * fraction:.1f}% of image",
        transform=axis.transAxes, ha="center", va="top", fontsize=9,
    )
    axis.axis("off")


def draw_patient_grid(axis, method):
    """Draw 80 easy and 20 hard patients with rounded false-negative counts."""
    easy_count, hard_count = 80, 20
    easy_misses = round(easy_count * RISKS[method]["easy"])
    hard_misses = round(hard_count * RISKS[method]["hard"])
    patient_types = np.array(["easy"] * easy_count + ["hard"] * hard_count)
    missed = np.zeros(100, dtype=bool)
    missed[:easy_misses] = True
    missed[easy_count:easy_count + hard_misses] = True

    # Put missed cases first within each group so their concentration is visible.
    x = np.tile(np.arange(10), 10)
    y = 9 - np.repeat(np.arange(10), 10)
    facecolors = [COLORS[kind] for kind in patient_types]
    axis.scatter(x, y, s=92, c=facecolors, edgecolors="white", linewidths=0.8)
    axis.scatter(x[missed], y[missed], s=55, marker="x", c=COLORS["miss"], linewidths=2)
    axis.axhline(1.5, color="white", linewidth=4)
    axis.text(9.8, 5.5, "80 easy", color=COLORS["easy"], ha="left", va="center", fontweight="bold")
    axis.text(9.8, 0.5, "20 hard", color=COLORS["hard"], ha="left", va="center", fontweight="bold")
    marginal = EASY_SHARE * RISKS[method]["easy"] + (1 - EASY_SHARE) * RISKS[method]["hard"]
    axis.set_title(method, fontsize=15, fontweight="bold", pad=8,
                   color=COLORS["crc"] if method == "Global CRC" else COLORS["recirc"])
    axis.text(
        0.5, -0.08,
        f"Easy FNR {RISKS[method]['easy']:.0%}  |  Hard FNR {RISKS[method]['hard']:.0%}  |  Global {marginal:.1%}",
        transform=axis.transAxes, ha="center", va="top", fontsize=10, fontweight="bold",
    )
    axis.set_xlim(-0.7, 12.7)
    axis.set_ylim(-0.8, 9.8)
    axis.set_aspect("equal")
    axis.axis("off")


def make_part_a(easy, hard):
    figure = plt.figure(figsize=(13.5, 9.3), facecolor="white")
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.34, wspace=0.20)
    show_overlay(figure.add_subplot(grid[0, 0]), easy, "Easy patient", "large, conspicuous lesion", COLORS["easy"])
    show_overlay(figure.add_subplot(grid[0, 1]), hard, "Hard patient", "small, subtle lesion", COLORS["hard"])
    draw_patient_grid(figure.add_subplot(grid[1, 0]), "Global CRC")
    draw_patient_grid(figure.add_subplot(grid[1, 1]), "ReCIRC")

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["easy"], markeredgecolor="white", markersize=10, label="easy patient"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["hard"], markeredgecolor="white", markersize=10, label="hard patient"),
        Line2D([0], [0], marker="x", color=COLORS["miss"], markersize=9, linewidth=0, markeredgewidth=2, label="false negative (missed positive)"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035))
    figure.suptitle("Same global target, different concentration of risk", fontsize=21, fontweight="bold", y=0.985)
    figure.text(0.5, 0.93, r"Illustrative example with target $\alpha=0.10$", ha="center", fontsize=12, color="#555555")
    #figure.text(
        #0.5, 0.012,
        #"100-patient schematic: icon counts are rounded. Images provide context only; risk values are hypothetical.",
        #ha="center", fontsize=9, color="#666666",
    #)
    return figure


def load_experiment_3_bin_summary(path=EXPERIMENT_3_BIN_RESULTS):
    """Return mean bin risk and a normal-approximation 95% CI over trials."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Experiment 3 bin results not found: {path}\n"
            "Run experiments/run_experiment_3_tumor_segmentation.py first."
        )
    results = pd.read_csv(path)
    required = {"method", "bin", "trial", "conditional_risk"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    methods = ["Standard CRC", "Rectified CRC"]
    results = results.loc[results["method"].isin(methods)].copy()
    summary = (
        results.groupby(["method", "bin"], as_index=False)["conditional_risk"]
        .agg(mean="mean", std="std", n="count")
    )
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["n"])
    if summary.groupby("method")["bin"].nunique().ne(5).any():
        raise ValueError("Expected exactly five difficulty bins for each method.")
    return summary, int(results["trial"].nunique())


def make_part_b():
    """Compare empirical curves above selected, outlined easy/hard RGB examples."""
    summary, n_trials = load_experiment_3_bin_summary()
    figure = plt.figure(figsize=(13, 11), facecolor="white")
    grid = figure.add_gridspec(
        2, 5, height_ratios=[5.4, 3.0], width_ratios=[1, 1, 0.65, 1, 1],
        left=0.12, right=0.97, bottom=0.12, top=0.93, hspace=0.38, wspace=0.10,
    )
    for group, identifiers, columns in [
        ("easy", [1626, 1761], [0, 1]),
        ("hard", [1706, 1714], [3, 4]),
    ]:
        for identifier, column in zip(identifiers, columns):
            image_axis = figure.add_subplot(grid[1, column])
            image = _read_rgb(IMAGE_DIR / f"{identifier}.jpg")
            mask = plt.imread(IMAGE_DIR / f"{identifier}_gt_mask.jpg")
            if mask.ndim == 3:
                mask = mask[..., :3].mean(axis=2)
            cutoff = 127 if np.issubdtype(mask.dtype, np.integer) else 0.5
            mask = mask > cutoff
            if mask.shape != image.shape[:2]:
                raise ValueError(f"RGB/mask dimensions differ for image {identifier}")
            image_axis.imshow(image)
            image_axis.set_anchor("N")
            image_axis.contour(mask, levels=[0.5], colors=[COLORS[group]], linewidths=1.8)
            image_axis.axis("off")
        group_box = grid[1, columns[0]:columns[-1] + 1].get_position(figure)
        figure.text(
            (group_box.x0 + group_box.x1) / 2, group_box.y1 + 0.012,
            f"{group.capitalize()} examples", ha="center", va="bottom",
            color=COLORS[group], fontsize=15, fontweight="bold",
        )
    axis = figure.add_subplot(grid[0, :])
    axis.axhline(ALPHA, color="#333333", linestyle="--", linewidth=1.6)
    for method, color in [("Standard CRC", COLORS["crc"]),
                          ("Rectified CRC", COLORS["recirc"])]:
        values = summary.loc[summary["method"].eq(method)].sort_values("bin")
        bins = values["bin"].to_numpy() + 1
        risk = values["mean"].to_numpy()
        ci95 = values["ci95"].to_numpy()
        axis.fill_between(bins, risk - ci95, risk + ci95, color=color, alpha=0.18, linewidth=0)
        axis.errorbar(
            bins, risk, yerr=ci95, color=color, marker="o", markersize=8,
            linewidth=3.0, capsize=4, capthick=1.5, zorder=3,
        )
    axis.set_title(r"Same global target: $\alpha = 0.10$", fontsize=18, fontweight="bold", pad=18)
    axis.set_xlabel("Difficulty bin", fontsize=15)
    axis.set_ylabel("False-negative risk", fontsize=15)
    axis.set_xlim(0.7, 5.3)
    axis.set_ylim(0, 0.46)
    axis.set_xticks(np.arange(1, 6), ["1\nEasiest", "2", "3", "4", "5\nHardest"])
    axis.tick_params(axis="both", labelsize=13)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.text(3.55, 0.205, "Global CRC", color=COLORS["crc"], fontsize=14, fontweight="bold")
    axis.text(3.30, 0.061, "ReCIRC", color=COLORS["recirc"], fontsize=14, fontweight="bold")
    #figure.text(
        #0.545, 0.025,
        #f"Experiment 3 · mean and 95% CI over {n_trials} trials · outlines show ground truth",
        #ha="center", fontsize=10, color="#555555",
    #)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = {}
    if args.part in {"a", "all"}:
        easy, hard = choose_image_examples()
        figures["part_a_same_target_two_patients"] = make_part_a(easy, hard)
    if args.part in {"b", "all"}:
        figures["part_b_flattening_risk_curve"] = make_part_b()
    for name, figure in figures.items():
        figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        figure.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(figure)
    print(f"Saved {len(figures)} figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
