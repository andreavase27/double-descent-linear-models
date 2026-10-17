"""Double descent on the synthetic dataset: least squares and ridge.

Run from the project root:  python src/run_synthetic.py
Figures are saved in figures/.
"""
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synthetic import generate_synthetic
from models import least_squares, ridge, ridge_cv
from experiment import run_experiment, summarize, theoretical_risk

# ---------------------------------------------------------------- settings
N_TRAIN = 100      # training points: fixed, as required by the assignment
N_TEST = 2000      # test points: large, so the test error is a precise estimate
D_TOTAL = 500      # features available; the model uses the first d of them
SIGMA = 0.5        # label noise standard deviation (noise variance 0.25)
N_REPS = 50        # independent training sets (seeds 0..49)

# Model sizes: coarse far from the threshold, dense around d = n
D_VALUES = sorted(set(
    list(range(5, 80, 5))
    + list(range(80, 121, 2))
    + list(range(130, 201, 10))
    + list(range(225, D_TOTAL + 1, 25))
))

SCENARIOS = {
    "A": {"decay": 0.0, "label": "equally important features"},
    "B": {"decay": 1.0, "label": "few important features"},
}

# Penalty grid for cross-validation, from almost no penalty to very strong
LAMBDAS = np.logspace(-3, 3, 13)

# Models compared in the ridge figure: name -> training function (X, y) -> w
MODELS = {
    "least squares (min-norm)": least_squares,
    "ridge, λ = 1": partial(ridge, lam=1.0),
    "ridge, λ = 10": partial(ridge, lam=10.0),
    "ridge, λ = 100": partial(ridge, lam=100.0),
    "ridge, λ tuned by 5-fold CV": partial(ridge_cv, lambdas=LAMBDAS),
}

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Colors (validated categorical palette, fixed order) and text inks
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEST_COLOR = PALETTE[0]
TRAIN_COLOR = PALETTE[1]
INK = "#52514e"
MUTED = "#8a8984"


def run_scenario(decay, fit):
    """Run one training algorithm for every d on N_REPS synthetic datasets."""
    def sample_data(rng):
        X_tr, y_tr, X_te, y_te, _ = generate_synthetic(
            N_TRAIN, N_TEST, D_TOTAL, SIGMA, rng, decay=decay
        )
        return X_tr, y_tr, X_te, y_te

    return run_experiment(sample_data, D_VALUES, N_REPS, fit)


def expected_signal_weights(decay):
    """Expected share of signal variance carried by each feature."""
    w = np.arange(1, D_TOTAL + 1) ** (-2.0 * decay)
    return w / w.sum()


def plot_scenario(name, results, decay, label):
    """Train error, test error (with theory) and norm of w for least squares."""
    d = np.array(D_VALUES)
    fig, (ax_err, ax_norm) = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: train and test error
    for key, color, text in [("test", TEST_COLOR, "test error"),
                             ("train", TRAIN_COLOR, "train error")]:
        med, q25, q75 = summarize(results[key])
        ax_err.fill_between(d, q25, q75, color=color, alpha=0.2, linewidth=0)
        ax_err.plot(d, med, color=color, linewidth=2,
                    label=f"{text} (median, band = IQR)")

    weights = expected_signal_weights(decay)
    theory = np.array([theoretical_risk(k, N_TRAIN, SIGMA, weights) for k in d])
    finite = np.isfinite(theory)
    ax_err.plot(d[finite], theory[finite], color=INK, linewidth=1,
                linestyle="--", label="test error (theory, mean)")

    ax_err.axhline(SIGMA ** 2, color=MUTED, linewidth=1, linestyle=":")
    ax_err.text(D_TOTAL, SIGMA ** 2, "noise floor", color=INK,
                ha="right", va="bottom", fontsize=8)
    ax_err.axhline(1 + SIGMA ** 2, color=MUTED, linewidth=1, linestyle=":")
    ax_err.text(D_TOTAL, 1 + SIGMA ** 2, "predict 0", color=INK,
                ha="right", va="bottom", fontsize=8)

    # symlog: logarithmic above 0.1, linear below, so train error 0 is visible
    ax_err.set_yscale("symlog", linthresh=0.1, linscale=0.5)
    ax_err.set_ylim(0, 3000)
    ax_err.set_ylabel("mean squared error")

    # Right panel: norm of the coefficients
    med, q25, q75 = summarize(results["norm"])
    ax_norm.fill_between(d, q25, q75, color=TEST_COLOR, alpha=0.2, linewidth=0)
    ax_norm.plot(d, med, color=TEST_COLOR, linewidth=2,
                 label="norm of w (median, band = IQR)")
    ax_norm.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax_norm.text(D_TOTAL, 1.0, r"$\|\beta\| = 1$", color=INK,
                 ha="right", va="bottom", fontsize=8)
    ax_norm.set_yscale("log")
    ax_norm.set_ylabel(r"$\|w\|$")

    for ax in (ax_err, ax_norm):
        ax.axvline(N_TRAIN, color=INK, linewidth=1)
        ax.set_xlabel("number of features d")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax_err.text(N_TRAIN + 5, 1500, "d = n", color=INK, fontsize=8)

    fig.suptitle(f"Scenario {name}: {label} "
                 f"(n = {N_TRAIN}, $\\sigma$ = {SIGMA}, {N_REPS} repetitions)")
    fig.tight_layout()
    save(fig, f"synthetic_ls_{name}")


def plot_ridge_comparison(name, all_results, label):
    """Test error and norm of w for least squares and ridge."""
    d = np.array(D_VALUES)
    fig, (ax_err, ax_norm) = plt.subplots(1, 2, figsize=(10, 4))

    for color, (model, results) in zip(PALETTE, all_results.items()):
        for ax, key in ((ax_err, "test"), (ax_norm, "norm")):
            med, q25, q75 = summarize(results[key])
            ax.fill_between(d, q25, q75, color=color, alpha=0.15, linewidth=0)
            ax.plot(d, med, color=color, linewidth=2, label=model)

    ax_err.axhline(SIGMA ** 2, color=MUTED, linewidth=1, linestyle=":")
    ax_err.text(D_TOTAL, SIGMA ** 2, "noise floor", color=INK,
                ha="right", va="bottom", fontsize=8)
    ax_err.axhline(1 + SIGMA ** 2, color=MUTED, linewidth=1, linestyle=":")
    ax_err.text(D_TOTAL, 1 + SIGMA ** 2, "predict 0", color=INK,
                ha="right", va="bottom", fontsize=8)
    ax_err.set_ylabel("test error (median, band = IQR)")
    ax_norm.set_ylabel(r"$\|w\|$ (median, band = IQR)")

    for ax in (ax_err, ax_norm):
        ax.set_yscale("log")
        ax.axvline(N_TRAIN, color=INK, linewidth=1)
        ax.set_xlabel("number of features d")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
    ax_err.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(f"Scenario {name}: {label}, least squares vs ridge "
                 f"(n = {N_TRAIN}, $\\sigma$ = {SIGMA}, {N_REPS} repetitions)")
    fig.tight_layout()
    save(fig, f"synthetic_ridge_{name}")


def save(fig, filename):
    """Save a figure as PDF (for the report) and PNG (for quick viewing)."""
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{filename}.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for name, cfg in SCENARIOS.items():
        all_results = {}
        for model, fit in MODELS.items():
            all_results[model] = run_scenario(cfg["decay"], fit)

            med_test = summarize(all_results[model]["test"])[0]
            best = int(np.argmin(med_test))
            print(f"Scenario {name} | {model:40s} | "
                  f"peak at d = {D_VALUES[int(np.argmax(med_test))]:3d}, "
                  f"max {med_test.max():8.2f} | "
                  f"best d = {D_VALUES[best]:3d}, min {med_test[best]:.3f}")

        ls_results = all_results["least squares (min-norm)"]
        plot_scenario(name, ls_results, cfg["decay"], cfg["label"])
        plot_ridge_comparison(name, all_results, cfg["label"])