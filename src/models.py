import numpy as np


def least_squares(X, y):
    """Least squares coefficients; minimum-norm solution when d > n.

    X has shape (n, d), y has shape (n,). Returns w of shape (d,).
    """
    n, d = X.shape

    if d <= n:
        # Classical regime: solve the normal equations (X^T X) w = X^T y
        w = np.linalg.solve(X.T @ X, X.T @ y)
    else:
        # Overparameterized regime: minimum-norm interpolant
        # w = X^T a, where a solves (X X^T) a = y
        a = np.linalg.solve(X @ X.T, y)
        w = X.T @ a

    return w


def predict(X, w):
    """Predictions of the linear model with coefficients w."""
    return X @ w


def mean_squared_error(y_true, y_pred):
    """Average squared difference between labels and predictions."""
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    # Sanity checks, run only when this file is executed directly
    from synthetic import generate_synthetic

    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test, beta = generate_synthetic(
        n_train=100, n_test=2000, D=500, sigma=0.5, rng=rng
    )

    for d in [20, 50, 90, 100, 110, 200, 500]:
        Xd_train = X_train[:, :d]
        Xd_test = X_test[:, :d]

        w = least_squares(Xd_train, y_train)

        train_err = mean_squared_error(y_train, predict(Xd_train, w))
        test_err = mean_squared_error(y_test, predict(Xd_test, w))

        # Reference solution from numpy's pseudoinverse, used only as a check
        w_ref = np.linalg.pinv(Xd_train) @ y_train
        gap = np.max(np.abs(w - w_ref))

        print(
            f"d={d:3d}  train={train_err:10.4f}  test={test_err:10.4f}  "
            f"norm={np.linalg.norm(w):8.3f}  gap_vs_pinv={gap:.1e}"
        )