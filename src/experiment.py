import numpy as np

from models import least_squares, predict, mean_squared_error


def run_experiment(sample_data, d_values, n_reps, fit=least_squares):
    """Train one model per dimension d, repeated over independent datasets.

    sample_data : function rng -> (X_train, y_train, X_test, y_test).
                  The columns must be ordered so that the model of size d
                  uses the first d of them.
    d_values    : list of model sizes d to try
    n_reps      : number of repetitions; repetition r uses seed r
    fit         : function (X, y) -> w, the training algorithm

    Returns a dict of arrays of shape (n_reps, len(d_values)) with the
    train error, the test error and the norm of w for every run.
    """
    shape = (n_reps, len(d_values))
    results = {
        "train": np.zeros(shape),
        "test": np.zeros(shape),
        "norm": np.zeros(shape),
    }

    for r in range(n_reps):
        # A fresh training set (and test set) for every repetition
        rng = np.random.default_rng(r)
        X_train, y_train, X_test, y_test = sample_data(rng)

        for i, d in enumerate(d_values):
            # The model of size d sees only the first d features
            w = fit(X_train[:, :d], y_train)

            results["train"][r, i] = mean_squared_error(
                y_train, predict(X_train[:, :d], w)
            )
            results["test"][r, i] = mean_squared_error(
                y_test, predict(X_test[:, :d], w)
            )
            results["norm"][r, i] = np.linalg.norm(w)

    return results


def summarize(values):
    """Median and interquartile range across repetitions (axis 0)."""
    q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
    return median, q25, q75


def theoretical_risk(d, n, sigma, signal_weights):
    """Expected test error of min-norm least squares, Gaussian features.

    signal_weights[j] is the share of the signal variance carried by
    feature j (the weights sum to 1). Formulas from Belkin, Hsu and Xu,
    "Two models of double descent for weak features" (2020).
    """
    seen = signal_weights[:d].sum()      # signal the model can see
    unseen = 1.0 - seen                  # signal it cannot see
    effective_noise = sigma ** 2 + unseen

    if d < n - 1:
        # Classical regime: unseen signal + noise, amplified by variance
        return effective_noise * (1 + d / (n - d - 1))
    if d > n + 1:
        # Overparameterized regime: lost share of the seen signal + noise
        return seen * (1 - n / d) + effective_noise * (1 + n / (d - n - 1))
    # Around d = n the expected error is infinite
    return np.inf