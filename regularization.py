"""Practice 5: regularization as a cure for an ill-posed estimation problem.

Ordinary least squares on correlated, noisy features is ill-posed in the sense of
Hadamard: the solution is not stable under small perturbations of the data. Ridge
and Lasso restore stability by adding a penalty term. Your job is to build the six
estimators below, then write up the comparison (see README, "What to hand in").
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes

ALPHA_GRID = np.linspace(1e-3, 1e2, 300)
C_GRID = np.logspace(-3, 2, 60)


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Split into train/test, then standardise using training statistics only.

    The split comes first on purpose. Fitting the scaler on the full matrix would
    leak test-set means and variances into training - a small, silent error that
    inflates every score you report afterwards.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        list[np.ndarray]: [X_train, X_test, y_train, y_test], features standardised.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)

    return [scaler.transform(X_train), scaler.transform(X_test), y_train, y_test]


def get_regression_data() -> list[np.ndarray]:
    """Load and preprocess the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: [X_train, X_test, y_train, y_test].
    """
    data = load_diabetes()
    return preprocess(data.data, data.target)


def get_classification_data() -> list[np.ndarray]:
    """Load and preprocess the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: [X_train, X_test, y_train, y_test].
    """
    data = load_breast_cancer()
    return preprocess(data.data, data.target)


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit an unregularised least-squares model. This is your baseline.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        BaseEstimator: fitted LinearRegression.
    """
    return LinearRegression().fit(X, y)


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit a ridge (L2) model, choosing `alpha` from ALPHA_GRID with GridSearchCV.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        BaseEstimator: the refitted best estimator (`.best_estimator_`), so that
            the returned object exposes `.alpha` and `.coef_`.
    """
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": ALPHA_GRID},
        scoring="r2",
        cv=5,
        n_jobs=-1,
    ).fit(X, y)

    # `.best_estimator_` is already refit on the whole of X, y by GridSearchCV
    # (refit=True by default), so the caller gets a usable model and not a wrapper
    return search.best_estimator_


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit a lasso (L1) model, choosing `alpha` from ALPHA_GRID with GridSearchCV.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): target vector.

    Returns:
        BaseEstimator: the refitted best estimator, exposing `.alpha` and `.coef_`.
    """
    search = GridSearchCV(
        # The default 1000 iterations are not always enough at the small-alpha
        # end of ALPHA_GRID, where the problem is barely penalised at all
        Lasso(max_iter=100_000),
        param_grid={"alpha": ALPHA_GRID},
        scoring="r2",
        cv=5,
        n_jobs=-1,
    ).fit(X, y)

    return search.best_estimator_


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit logistic regression with no regularisation at all.

    Note: `penalty=` was deprecated in scikit-learn 1.8 and is removed in 1.10.
    Switch the penalty off with `C=np.inf` instead.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): binary target vector.

    Returns:
        BaseEstimator: fitted LogisticRegression.
    """
    # C=np.inf is the only way to switch the penalty off in scikit-learn >= 1.8;
    # passing penalty=None raises a DeprecationWarning, which this project's
    # pytest configuration promotes to an error
    return LogisticRegression(C=np.inf, max_iter=10_000).fit(X, y)


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit L2-penalised logistic regression, tuning `C` over C_GRID with GridSearchCV.

    Note: L2 is `l1_ratio=0` in scikit-learn >= 1.8 (not `penalty="l2"`).

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): binary target vector.

    Returns:
        BaseEstimator: the refitted best estimator, exposing `.C` and `.coef_`.
    """
    search = GridSearchCV(
        LogisticRegression(l1_ratio=0, max_iter=10_000),
        param_grid={"C": C_GRID},
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    ).fit(X, y)

    return search.best_estimator_


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit L1-penalised logistic regression, tuning `C` over C_GRID with GridSearchCV.

    Note: L1 is `l1_ratio=1` in scikit-learn >= 1.8, and it needs a solver that
    supports it - `liblinear` or `saga`. The default `lbfgs` raises ValueError.

    Args:
        X (np.ndarray): feature matrix.
        y (np.ndarray): binary target vector.

    Returns:
        BaseEstimator: the refitted best estimator, exposing `.C` and `.coef_`.
    """
    search = GridSearchCV(
        # lbfgs cannot handle an L1 term; liblinear solves the penalised problem
        # by coordinate descent and drives coefficients to exactly zero, which is
        # the property this exercise is about
        LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=10_000),
        param_grid={"C": C_GRID},
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    ).fit(X, y)

    return search.best_estimator_
