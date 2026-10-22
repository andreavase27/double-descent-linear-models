import numpy as np


def least_squares(X, y):
    """Least squares coefficients; minimum-norm solution when d > n."""
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
    return np.mean((y_true - y_pred) ** 2)

def ridge(X, y, lam):
    n, d = X.shape

    if d <= n:
        # Primal form: solve (X^T X + lam I) w = X^T y, a d x d system
        w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    else:
        # Dual form, same solution: w = X^T a, with (X X^T + lam I) a = y,
        # an n x n system (cheaper when d > n)
        a = np.linalg.solve(X @ X.T + lam * np.eye(n), y)
        w = X.T @ a

    return w


def ridge_cv(X, y, lambdas, n_folds=5):
    """Ridge with lam chosen by k-fold cross-validation on (X, y) only.

    The lam with the smallest average validation error is used to refit on all data. """
 
    n = X.shape[0]
    folds = np.array_split(np.arange(n), n_folds)

    cv_errors = []
    for lam in lambdas:
        fold_errors = []
        for val_idx in folds:
            train_idx = np.setdiff1d(np.arange(n), val_idx)
            w = ridge(X[train_idx], y[train_idx], lam)
            fold_errors.append(
                mean_squared_error(y[val_idx], predict(X[val_idx], w))
            )
        cv_errors.append(np.mean(fold_errors))

    best_lam = lambdas[int(np.argmin(cv_errors))]
    return ridge(X, y, best_lam)

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