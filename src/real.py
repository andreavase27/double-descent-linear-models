import numpy as np
from sklearn.datasets import fetch_california_housing


def load_california():
    """California Housing (20,640 districts, 8 numeric features).
    The target is the median house value of the district, in units of
    100,000 dollars.
    """
    data = fetch_california_housing()
    return data.data, data.target


def standardize(train, test):
    """Center and scale with the mean and std of the TRAINING data only."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    return (train - mean) / std, (test - mean) / std


def median_distance(X):
    """Median Euclidean distance between pairs of rows of X."""
    diff = X[:, None, :] - X[None, :, :]          # all pairwise differences
    dist = np.sqrt((diff ** 2).sum(axis=-1))      # pairwise distances
    upper = np.triu_indices(X.shape[0], k=1)      # each pair counted once
    return np.median(dist[upper])


def random_fourier_features(X, V):
    """Map X to [cos(X v_1), sin(X v_1), cos(X v_2), sin(X v_2), ...].

    V has one random frequency vector per column. The cos and sin of the
    same frequency are kept next to each other, so the first d columns
    always contain d / 2 complete pairs (d even).
    """
    Z = X @ V
    features = np.empty((X.shape[0], 2 * V.shape[1]))
    features[:, 0::2] = np.cos(Z)
    features[:, 1::2] = np.sin(Z)
    return features


def make_real_sampler(X_all, y_all, n_train, n_test, n_freq, scale=1.0):
    """Return a function rng -> (Phi_train, y_train, Phi_test, y_test).

    Every call draws a new random train/test split from the full dataset,
    standardizes it with training statistics only, and builds 2 * n_freq
    random Fourier features. The bandwidth is the median distance between
    training points (median heuristic), multiplied by `scale`.
    """
    def sample(rng):
        # 1. Disjoint random train and test sets
        idx = rng.permutation(len(y_all))
        train_idx = idx[:n_train]
        test_idx = idx[n_train:n_train + n_test]

        # 2. Standardize inputs and target using the training set only
        X_train, X_test = standardize(X_all[train_idx], X_all[test_idx])
        y_train, y_test = standardize(y_all[train_idx], y_all[test_idx])

        # 3. Random frequencies, with bandwidth from the training set only
        bandwidth = scale * median_distance(X_train)
        V = rng.standard_normal((X_all.shape[1], n_freq)) / bandwidth

        # 4. The same frequencies are applied to train and test
        return (random_fourier_features(X_train, V), y_train,
                random_fourier_features(X_test, V), y_test)

    return sample