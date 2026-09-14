"""Model-independent held-out risk calibration. Never fits models or consumes RNG."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bin_groups(bins, n_bins):
    """Preserve caller-defined zero-based bins, including empty ones."""
    bins = np.asarray(bins)
    return {f"Bin {b + 1}": bins == b for b in range(n_bins)}


def calibration_rows(budgets, losses, groups=None, a_hat=np.nan):
    """Loss matrix shape: observations x budgets. Groups are fixed boolean masks."""
    budgets = np.asarray(budgets, dtype=float)
    losses = np.asarray(losses)
    if budgets.ndim != 1 or not len(budgets) or not np.isfinite(budgets).all() or np.any(np.diff(budgets) <= 0):
        raise ValueError("budgets must be finite and strictly increasing")
    if losses.ndim != 2 or losses.shape[1] != len(budgets) or not len(losses):
        raise ValueError("losses must have shape (observations, budgets), with observations > 0")
    if losses.dtype.kind not in "bifu" or not np.isfinite(losses).all():
        raise ValueError("losses must be finite")
    masks = {"Overall": np.ones(len(losses), dtype=bool)}
    for name, mask in (groups or {}).items():
        mask = np.asarray(mask)
        if name == "Overall" or mask.dtype != bool or mask.shape != (len(losses),):
            raise ValueError("groups require boolean observation masks; Overall is reserved")
        masks[name] = mask
    rows = []
    for j, a in enumerate(budgets):
        for group, mask in masks.items():
            values = losses[mask, j]
            n = len(values)
            risk = float(values.mean()) if n else np.nan
            rows.append({"group": group, "a": float(a), "n": n,
                         "risk": risk, "gap": risk - a, "a_hat": a_hat,
                         "se": float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan})
    return pd.DataFrame(rows)


class RiskCalibration:
    """Collect one fitted ReCIRC family's diagnostic per trial, then export."""
    def __init__(self, ylabel="Average loss", operating_range=(0.05, 0.20), target=0.10):
        self.ylabel = ylabel
        self.operating_range = operating_range
        self.target = target
        self.frames = []

    def add_trial(self, budgets, losses, groups=None, trial=None, seed=None, a_hat=np.nan):
        trial = len(self.frames) if trial is None else trial
        if any(frame["trial"].iloc[0] == trial for frame in self.frames):
            raise ValueError("duplicate diagnostic trial")
        frame = calibration_rows(budgets, losses, groups, a_hat)
        if self.frames:
            first = self.frames[0]
            if list(frame.group.unique()) != list(first.group.unique()) or not np.array_equal(frame.a.unique(), first.a.unique()):
                raise ValueError("all trials must use the same budget grid and group names")
        frame["trial"], frame["seed"] = trial, seed
        self.frames.append(frame)
        return frame

    def save(self, output_dir, make_plot=True):
        if self.frames:
            save_risk_calibration(pd.concat(self.frames, ignore_index=True), output_dir,
                                  make_plot=make_plot, ylabel=self.ylabel,
                                  operating_range=self.operating_range, target=self.target)


def operating_range_summary(summary, operating_range=(0.05, 0.20)):
    """Summarize the plotted mean gap inside a prespecified budget range."""
    lower, upper = map(float, operating_range)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("operating_range must contain two finite increasing values")
    rows = []
    for group, values in summary.groupby("group", sort=False):
        values = values.loc[values["a"].between(lower, upper)].copy()
        values["signed_deviation"] = values["risk"] - values["a"]
        finite = values.loc[np.isfinite(values["signed_deviation"])]
        if finite.empty:
            maximum = average = budget = observed = np.nan
            trials = 0
        else:
            maximum_row = finite.loc[finite["signed_deviation"].idxmax()]
            maximum = float(maximum_row["signed_deviation"])
            average = float(finite["signed_deviation"].mean())
            budget = float(maximum_row["a"])
            observed = float(maximum_row["risk"])
            trials = int(finite["trials"].min())
        rows.append({
            "group": group,
            "operating_range_min": lower,
            "operating_range_max": upper,
            "max_signed_deviation": maximum,
            "a_at_max_signed_deviation": budget,
            "observed_loss_at_max_signed_deviation": observed,
            "average_signed_deviation": average,
            "budgets_in_range": int(len(finite)),
            "trials": trials,
        })
    return pd.DataFrame(rows)


def save_risk_calibration(curves, output_dir, make_plot=True, ylabel="Average loss",
                          operating_range=(0.05, 0.20), target=0.10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "risk_calibration_per_trial.csv", index=False)
    summary = (curves.groupby(["group", "a"], sort=False, as_index=False)
               .agg(risk=("risk", "mean"), trials=("risk", "count")))
    summary.to_csv(output_dir / "risk_calibration_summary.csv", index=False)
    operating = operating_range_summary(summary, operating_range)
    operating.to_csv(output_dir / "risk_calibration_operating_range.csv", index=False)
    if not make_plot:
        return
    lower, upper = map(float, operating_range)
    groups = list(curves["group"].unique())
    n_rows = int(np.ceil(len(groups) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, max(7, 3.5 * n_rows)),
                             squeeze=False, sharex=True, sharey=True)
    for ax, group in zip(axes.flat, groups):
        sub = curves[curves["group"] == group]
        ax.axvspan(lower, upper, color="#F2C94C", alpha=0.25,
                   label=f"Operating range [{lower:.2f}, {upper:.2f}]")
        for _, trial in sub.groupby("trial"):
            ax.plot(trial["a"], trial["risk"] - trial["a"], color="#002F6C", alpha=0.18, lw=1)
        mean = summary[summary["group"] == group]
        ax.plot(mean["a"], mean["risk"] - mean["a"], color="#002F6C", lw=2, label="Mean loss − a")
        ax.axhline(0, color="black", linestyle="--", lw=1, label="Loss = budget")
        if target is not None:
            ax.axvline(target, color="#C0392B", linestyle="-.", lw=1.2,
                       label=fr"Target $\alpha={target:.2f}$")
        operating_row = operating.loc[operating["group"] == group].iloc[0]
        if np.isfinite(operating_row["max_signed_deviation"]):
            budget = operating_row["a_at_max_signed_deviation"]
            maximum = operating_row["max_signed_deviation"]
            ax.scatter([budget], [maximum], color="#8A6D00", s=22, zorder=5)
            ax.annotate(
                f"Max signed gap: {maximum:+.3f}\nat $a={budget:.3f}$",
                xy=(budget, maximum), xytext=(0.04, 0.94), textcoords="axes fraction",
                ha="left", va="top", fontsize=7.5, color="#6F5700",
                arrowprops={"arrowstyle": "-", "color": "#8A6D00", "lw": 0.8},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )
        n_min, n_max = int(sub["n"].min()), int(sub["n"].max())
        count_label = str(n_min) if n_min == n_max else f"{n_min}–{n_max}"
        ax.set(title=f"{group} (n={count_label})",
               xlabel="Risk budget a", ylabel=f"{ylabel} − a", xlim=(0, 1))
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7.5)
    count = curves["trial"].nunique()
    subtitle = ("Fixed groups within each trial; faint lines show individual trials" if count > 1
                else "Fixed groups within each trial | single-trial diagnostic")
    fig.suptitle(f"ReCIRC held-out risk calibration | {count} trial(s)\n{subtitle}")
    fig.tight_layout()
    fig.savefig(output_dir / "risk_calibration.png", dpi=200)
    fig.savefig(output_dir / "risk_calibration.pdf")
    plt.close(fig)
