"""Extension: gradient descent vs closed-form least squares (synthetic data)."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Larger text in all figures, so they stay readable when scaled in the report
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

from synthetic import generate_synthetic
from models import least_squares, gradient_descent, predict, mean_squared_error
from experiment import summarize

# ---------------------------------------------------------------- settings
N_TRAIN = 100
N_TEST = 2000
D_TOTAL = 500
SIGMA = 0.5
DECAY = 0.0              # scenario A
N_REPS = 20              # fewer repetitions: gradient descent is slower
ITERATIONS = [100, 1000, 10000]

D_VALUES = sorted(set(
    list(range(10, 80, 10))
    + list(range(80, 121, 4))
    + list(range(140, 201, 20))
    + list(range(250, D_TOTAL + 1, 50))
))

FIGURES = Path(__file__).resolve().parent.parent / "figures"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#52514e"


def run():
    """Test error and distance from the closed form, for every run."""
    shape = (N_REPS, len(D_VALUES))
    test = {"closed form": np.zeros(shape)}
    dist = {}
    for t in ITERATIONS:
        test[f"GD, {t} iterations"] = np.zeros(shape)
        dist[f"GD, {t} iterations"] = np.zeros(shape)

    for r in range(N_REPS):
        rng = np.random.default_rng(r)
        X_train, y_train, X_test, y_test, _ = generate_synthetic(
            N_TRAIN, N_TEST, D_TOTAL, SIGMA, rng, decay=DECAY
        )
        for i, d in enumerate(D_VALUES):
            Xd_train, Xd_test = X_train[:, :d], X_test[:, :d]

            w_ls = least_squares(Xd_train, y_train)
            test["closed form"][r, i] = mean_squared_error(
                y_test, predict(Xd_test, w_ls))

            for t in ITERATIONS:
                key = f"GD, {t} iterations"
                w_gd = gradient_descent(Xd_train, y_train, t)
                test[key][r, i] = mean_squared_error(
                    y_test, predict(Xd_test, w_gd))
                dist[key][r, i] = (np.linalg.norm(w_gd - w_ls)
                                   / np.linalg.norm(w_ls))
    return test, dist


def plot(test, dist):
    d = np.array(D_VALUES)
    fig, (ax_err, ax_dist) = plt.subplots(1, 2, figsize=(10, 4))

    for color, (label, values) in zip(PALETTE, test.items()):
        med, q25, q75 = summarize(values)
        ax_err.fill_between(d, q25, q75, color=color, alpha=0.15, linewidth=0)
        ax_err.plot(d, med, color=color, linewidth=2, label=label)
    ax_err.set_ylabel("test error (median, band = IQR)")

    for color, (label, values) in zip(PALETTE[1:], dist.items()):
        med, q25, q75 = summarize(values)
        ax_dist.fill_between(d, q25, q75, color=color, alpha=0.15, linewidth=0)
        ax_dist.plot(d, med, color=color, linewidth=2, label=label)
    ax_dist.set_ylabel(r"$\|w_{GD} - w_{LS}\| \, / \, \|w_{LS}\|$")

    for ax in (ax_err, ax_dist):
        ax.set_yscale("log")
        ax.axvline(N_TRAIN, color=INK, linewidth=1)
        ax.set_xlabel("number of features d")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        # One legend is enough: both panels use the same colors
    ax_err.legend(frameon=False, fontsize=11, loc="upper right")

    fig.suptitle(f"Gradient descent from w = 0 vs closed form, scenario A "
                 f"(n = {N_TRAIN}, $\\sigma$ = {SIGMA}, {N_REPS} repetitions)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "synthetic_gd.pdf")
    plt.close(fig)


if __name__ == "__main__":
    test, dist = run()
    plot(test, dist)

    # Median relative distance from the closed form at a few model sizes
    for label, values in dist.items():
        med = summarize(values)[0]
        cells = "  ".join(f"d={d}: {med[D_VALUES.index(d)]:.1e}"
                          for d in [50, 100, 200, 500])
        print(f"{label:22s} | {cells}")