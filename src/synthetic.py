import numpy as np


def generate_synthetic(n_train, n_test, D, sigma, rng, decay=0.0):
    """Generate a synthetic regression dataset with noisy linear labels.

    Model: y = x . beta + sigma * eps, with x ~ N(0, I_D) and eps ~ N(0, 1).

    decay controls how the signal is spread across features: the j-th
    coefficient is scaled by j^(-decay) before normalisation.
    decay = 0: all features equally important (on average).
    decay > 0: the first features carry most of the signal.
    """
    # True coefficient vector, rescaled to unit norm
    beta = rng.standard_normal(D) * np.arange(1, D + 1) ** (-decay)
    beta = beta / np.linalg.norm(beta)

    # Inputs: independent standard Gaussian entries
    X_train = rng.standard_normal((n_train, D))
    X_test = rng.standard_normal((n_test, D))

    # Labels: linear signal plus Gaussian noise
    y_train = X_train @ beta + sigma * rng.standard_normal(n_train)
    y_test = X_test @ beta + sigma * rng.standard_normal(n_test)

    return X_train, y_train, X_test, y_test, beta


if __name__ == "__main__":
    # Sanity checks, run only when this file is executed directly
    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test, beta = generate_synthetic(
        n_train=100, n_test=2000, D=500, sigma=0.5, rng=rng
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("norm of beta: ", np.linalg.norm(beta))
    print("variance of y_test:", y_test.var(), "(expected about 1.25)")

    # Same seed must give the same data
    rng_again = np.random.default_rng(0)
    X_again = generate_synthetic(100, 2000, 500, 0.5, rng_again)[0]
    print("reproducible:", np.array_equal(X_train, X_again))