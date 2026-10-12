"""Portfolio objective evaluation.

Three objectives, all in the maximization convention:

  f1(x) = -x^T Σ x   (negative risk)
  f2(x) = μ^T x       (expected return)
  f3(x) = x^T D x     (diversification)

The evaluator always uses the *full* covariance matrix Σ, regardless of
whether a sparse approximation was used during optimization.
"""

from __future__ import annotations

import numpy as np

from qportfolio.utils.logging import get_logger
from qportfolio.utils.validation import validate_array_finite, validate_symmetric

_log = get_logger("problems.objectives")


class ObjectiveEvaluator:
    """Evaluates the three portfolio objectives for binary selection vectors.

    Parameters
    ----------
    mu:
        Shape (N,) annualized expected returns.
    sigma:
        Shape (N, N) annualized covariance matrix (full).
    D:
        Shape (N, N) sector diversification matrix.
    """

    OBJECTIVE_NAMES = ("negative_risk", "expected_return", "diversification")

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        D: np.ndarray,
    ) -> None:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)

        validate_array_finite(mu, "mu")
        validate_array_finite(sigma, "sigma")
        validate_array_finite(D, "D")
        validate_symmetric(sigma, "sigma")

        N = mu.shape[0]
        if sigma.shape != (N, N):
            raise ValueError(f"sigma.shape={sigma.shape} inconsistent with mu.shape={mu.shape}")
        if D.shape != (N, N):
            raise ValueError(f"D.shape={D.shape} inconsistent with mu.shape={mu.shape}")

        self._mu = mu
        self._sigma = sigma
        self._D = D
        self._N = N

    @property
    def N(self) -> int:
        """Number of assets."""
        return self._N

    # ------------------------------------------------------------------
    # Single-portfolio evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> tuple[float, float, float]:
        """Evaluate objectives for a single portfolio vector x.

        Parameters
        ----------
        x:
            Shape (N,) binary {0,1} array (not validated for feasibility).

        Returns
        -------
        (f1, f2, f3) as Python floats.
        """
        x = np.asarray(x, dtype=np.float64)
        f1 = float(-x @ self._sigma @ x)
        f2 = float(self._mu @ x)
        f3 = float(x @ self._D @ x)
        return f1, f2, f3

    def evaluate_individual(self, x: np.ndarray, objective: int) -> float:
        """Evaluate a single objective (0=f1, 1=f2, 2=f3). For diagnostics."""
        if objective not in (0, 1, 2):
            raise ValueError(f"objective must be 0, 1, or 2; got {objective}")
        return self.evaluate(x)[objective]

    # ------------------------------------------------------------------
    # Batch evaluation (vectorized)
    # ------------------------------------------------------------------

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate objectives for a batch of portfolios.

        Parameters
        ----------
        X:
            Shape (M, N) binary matrix.

        Returns
        -------
        np.ndarray
            Shape (M, 3) float64 array of [f1, f2, f3] per row.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._N:
            raise ValueError(f"X.shape[1]={X.shape[1]}, expected N={self._N}")

        # f1 = -diag(X Σ X^T) = -(X @ sigma * X).sum(axis=1)
        f1 = -(X @ self._sigma * X).sum(axis=1)
        # f2 = X @ mu
        f2 = X @ self._mu
        # f3 = diag(X D X^T) = (X @ D * X).sum(axis=1)
        f3 = (X @ self._D * X).sum(axis=1)

        return np.column_stack([f1, f2, f3])

    def objective_names(self) -> list[str]:
        """Return the names of the three objectives."""
        return list(self.OBJECTIVE_NAMES)
