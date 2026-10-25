"""Extension: effect of the label noise on double descent (synthetic data)."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synthetic import generate_synthetic
from models import least_squares
from experiment import run_experiment, summarize, theoretical_risk

# ---------------------------------------------------------------- settings
N_TRAIN = 100
N_TEST = 2000
D_TOTAL = 500
N_REPS = 50
SIGMAS = [0.0, 0.25, 0.5, 1.0]     # label noise standard deviations

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

FIGURES = Path(__file__).resolve().parent.parent / "figures"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#52514e"


def expected_signal_weights(decay):
    """Expected share of signal variance carried by each feature."""
    w = np.arange(1, D_TOTAL + 1) ** (-2.0 * decay)
    return w / w.sum()


def run_scenario(decay, sigma):
    def sample_data(rng):
        X_tr, y_tr, X_te, y_te, _ = generate_synthetic(
            N_TRAIN, N_TEST, D_TOTAL, sigma, rng, decay=decay
        )
        return X_tr, y_tr, X_te, y_te

    return run_experiment(sample_data, D_VALUES, N_REPS, least_squares)


if __name__ == "__main__":
    d = np.array(D_VALUES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (name, cfg) in zip(axes, SCENARIOS.items()):
        weights = expected_signal_weights(cfg["decay"])

        for color, sigma in zip(PALETTE, SIGMAS):
            results = run_scenario(cfg["decay"], sigma)
            med, q25, q75 = summarize(results["test"])
            ax.fill_between(d, q25, q75, color=color, alpha=0.15, linewidth=0)
            ax.plot(d, med, color=color, linewidth=2, label=f"σ = {sigma}")

            # Theory (expected error) for the same noise level
            theory = np.array([theoretical_risk(k, N_TRAIN, sigma, weights)
                               for k in d])
            finite = np.isfinite(theory)
            ax.plot(d[finite], theory[finite], color=color, linewidth=1,
                    linestyle="--")

            peak = med[D_VALUES.index(N_TRAIN)]
            best = int(np.argmin(med))
            print(f"Scenario {name}, sigma = {sigma:4.2f} | "
                  f"error at d = n: {peak:9.2f} | "
                  f"best d = {D_VALUES[best]:3d}, min {med[best]:.3f}")

        ax.set_yscale("log")
        ax.axvline(N_TRAIN, color=INK, linewidth=1)
        ax.set_xlabel("number of features d")
        ax.set_ylabel("test error (median, band = IQR)")
        ax.set_title(f"Scenario {name}: {cfg['label']}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        ax.legend(frameon=False, fontsize=8, loc="upper right",
                  title="solid: simulation, dashed: theory", title_fontsize=7)

    fig.suptitle(f"Effect of the label noise, least squares "
                 f"(n = {N_TRAIN}, {N_REPS} repetitions)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "synthetic_noise.pdf")
    plt.close(fig)