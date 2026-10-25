"""Double descent on California Housing with random Fourier features."""
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from real import load_california, make_real_sampler
from models import least_squares, ridge, ridge_cv
from experiment import run_experiment, summarize

# settings
N_TRAIN = 100      # training points: same n as the synthetic experiments
N_TEST = 2000      # test points, drawn from the remaining districts
N_FREQ = 1000      # random frequencies -> up to 2 * N_FREQ = 2000 features
N_REPS = 50        # random train/test splits (seeds from 0 to 49)

# Model sizes (always even: each frequency gives a cos and a sin column)
D_VALUES = sorted(set(
    list(range(10, 80, 10))
    + list(range(80, 121, 2))
    + list(range(130, 201, 10))
    + [250, 300, 400, 500, 750, 1000, 1500, 2 * N_FREQ]
))

# Bandwidth = SCALE x median distance between training points
MAIN_SCALE = 1.0
SCALES = [1.0, 0.5, 0.25]

LAMBDAS = np.logspace(-3, 3, 13)
MODELS = {
    "least squares (min-norm)": least_squares,
    "ridge, λ = 1": partial(ridge, lam=1.0),
    "ridge, λ = 10": partial(ridge, lam=10.0),
    "ridge, λ = 100": partial(ridge, lam=100.0),
    "ridge, λ tuned by 5-fold CV": partial(ridge_cv, lambdas=LAMBDAS),
}

FIGURES = Path(__file__).resolve().parent.parent / "figures"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#52514e"
MUTED = "#8a8984"


def style(ax, ylabel):
    """Common axis styling: log scale, threshold line, light grid."""
    ax.set_yscale("log")
    ax.axvline(N_TRAIN, color=INK, linewidth=1)
    ax.set_xlabel("number of random features d")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)


def mark_mean_predictor(ax):
    """Targets are standardized, so predicting the training mean gives ~1."""
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax.text(D_VALUES[-1], 1.0, "predict mean", color=INK,
            ha="right", va="bottom", fontsize=8)


def plot_curves(ax, d, curves, key):
    """One line (median) and band (IQR) per entry of curves."""
    for color, (label, results) in zip(PALETTE, curves.items()):
        med, q25, q75 = summarize(results[key])
        ax.fill_between(d, q25, q75, color=color, alpha=0.15, linewidth=0)
        ax.plot(d, med, color=color, linewidth=2, label=label)


def save(fig, filename):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{filename}.pdf")
    plt.close(fig)


def plot_least_squares(results):
    d = np.array(D_VALUES)
    fig, (ax_err, ax_norm) = plt.subplots(1, 2, figsize=(10, 4))

    plot_curves(ax_err, d, {"test error": results}, "test")
    med, q25, q75 = summarize(results["train"])
    ax_err.fill_between(d, q25, q75, color=PALETTE[1], alpha=0.15, linewidth=0)
    ax_err.plot(d, med, color=PALETTE[1], linewidth=2, label="train error")
    mark_mean_predictor(ax_err)
    style(ax_err, "mean squared error (median, band = IQR)")
    # train error is 0 for d >= n: symlog shows it at the bottom
    ax_err.set_yscale("symlog", linthresh=0.1, linscale=0.5)
    ax_err.set_ylim(0, 1e4)
    ax_err.legend(frameon=False, fontsize=8, loc="upper right")

    plot_curves(ax_norm, d, {"norm of w": results}, "norm")
    style(ax_norm, r"$\|w\|$ (median, band = IQR)")
    ax_norm.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(f"California Housing, least squares on random Fourier "
                 f"features (n = {N_TRAIN}, {N_REPS} splits)")
    fig.tight_layout()
    save(fig, "real_ls")


def plot_ridge(all_results):
    d = np.array(D_VALUES)
    fig, (ax_err, ax_norm) = plt.subplots(1, 2, figsize=(10, 4))
    plot_curves(ax_err, d, all_results, "test")
    plot_curves(ax_norm, d, all_results, "norm")
    mark_mean_predictor(ax_err)
    style(ax_err, "test error (median, band = IQR)")
    style(ax_norm, r"$\|w\|$ (median, band = IQR)")
    ax_err.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"California Housing, least squares vs ridge "
                 f"(n = {N_TRAIN}, {N_REPS} splits)")
    fig.tight_layout()
    save(fig, "real_ridge")


def plot_bandwidth(scale_results):
    d = np.array(D_VALUES)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_curves(ax, d, scale_results, "test")
    mark_mean_predictor(ax)
    style(ax, "test error (median, band = IQR)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("California Housing, least squares: effect of the bandwidth",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "real_bandwidth")


def report(label, results):
    med_test = summarize(results["test"])[0]
    best = int(np.argmin(med_test))
    print(f"{label:40s} | peak at d = {D_VALUES[int(np.argmax(med_test))]:4d}, "
          f"max {med_test.max():9.2f} | "
          f"best d = {D_VALUES[best]:4d}, min {med_test[best]:.3f}")


if __name__ == "__main__":
    X_all, y_all = load_california()

    # 1. Least squares and ridge, bandwidth from the median heuristic
    sampler = make_real_sampler(X_all, y_all, N_TRAIN, N_TEST, N_FREQ,
                                scale=MAIN_SCALE)
    all_results = {}
    for model, fit in MODELS.items():
        all_results[model] = run_experiment(sampler, D_VALUES, N_REPS, fit)
        report(model, all_results[model])
    plot_least_squares(all_results["least squares (min-norm)"])
    plot_ridge(all_results)

    # 2. Least squares with narrower random features (smaller bandwidth)
    scale_results = {}
    for scale in SCALES:
        label = f"bandwidth = {scale} x median distance"
        sampler = make_real_sampler(X_all, y_all, N_TRAIN, N_TEST, N_FREQ,
                                    scale=scale)
        scale_results[label] = run_experiment(sampler, D_VALUES, N_REPS)
        report(label, scale_results[label])
    plot_bandwidth(scale_results)