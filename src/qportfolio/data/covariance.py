"""Covariance estimation with optional correlation-threshold sparsification.

Pipeline:
  Historical returns → sigma_full → (optional) sigma_sparse → report

The full covariance is always the reference for evaluating solutions.
The sparse covariance is an optional approximation for optimization
(especially quantum circuit construction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qportfolio.config import CovarianceConfig
from qportfolio.data.preprocessing import annualize_covariance
from qportfolio.utils.logging import get_logger

_log = get_logger("data.covariance")

# Grid of correlation thresholds to search during auto-selection
_THRESHOLD_GRID = np.arange(0.0, 1.0, 0.05)


class CovarianceEstimator:
    """Two-stage covariance estimator: full covariance, then optional sparsification.

    Parameters
    ----------
    config:
        Covariance estimation configuration.
    """

    def __init__(self, config: CovarianceConfig | None = None) -> None:
        self._config = config or CovarianceConfig()
        self._sigma_full: np.ndarray | None = None
        self._sigma_sparse: np.ndarray | None = None
        self._threshold: float | None = None
        self._retained_fraction: float | None = None
        self._report: dict | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, daily_returns: pd.DataFrame) -> CovarianceEstimator:
        """Compute the full (and optionally sparse) covariance matrix.

        Parameters
        ----------
        daily_returns:
            DataFrame of daily log returns. Must have no NaN values.

        Returns
        -------
        CovarianceEstimator
            Self (for chaining).
        """
        self._sigma_full = annualize_covariance(daily_returns)

        if self._config.method == "sparse":
            self._fit_sparse(daily_returns)
        else:
            self._report = self._build_report(self._sigma_full, None)

        return self

    def _fit_sparse(self, daily_returns: pd.DataFrame) -> None:
        """Chronological train/validation split → threshold selection → sparsification."""
        cfg = self._config
        n = len(daily_returns)
        split = int(n * (1 - cfg.validation_fraction))
        if split < 50:
            _log.warning("Validation split too small (%d rows). Falling back to full covariance.", split)
            self._sigma_sparse = self._sigma_full
            self._threshold = 0.0
            self._retained_fraction = 1.0
            assert self._sigma_full is not None
            self._report = self._build_report(self._sigma_full, 0.0)
            return

        train_returns = daily_returns.iloc[:split]
        val_returns = daily_returns.iloc[split:]

        sigma_train = annualize_covariance(train_returns)
        sigma_val = annualize_covariance(val_returns)

        # Determine threshold
        if cfg.threshold == "auto":
            tau = self._select_threshold(sigma_train, sigma_val, cfg)
        else:
            tau = float(cfg.threshold)

        sigma_sparse = self.sparsify(self._sigma_full, tau)  # type: ignore[arg-type]

        # PSD correction if needed
        is_psd_before, min_eig_before = self.check_psd(sigma_sparse)
        lam_used = 0.0
        if not is_psd_before:
            sigma_sparse, lam_used = self.apply_diagonal_loading(sigma_sparse, cfg.diagonal_loading)
            _log.info(
                "Applied diagonal loading λ=%.2e to sparse covariance for PSD correction.", lam_used
            )

        is_psd_after, min_eig_after = self.check_psd(sigma_sparse)

        self._sigma_sparse = sigma_sparse
        self._threshold = tau
        self._retained_fraction = self._compute_retained_fraction(self._sigma_full, tau)  # type: ignore[arg-type]
        self._report = {
            "threshold": tau,
            "retained_fraction": self._retained_fraction,
            "psd_before_loading": is_psd_before,
            "psd_after_loading": is_psd_after,
            "diagonal_loading_applied": lam_used > 0,
            "lam_used": lam_used,
            "min_eigenvalue_before": float(min_eig_before),
            "min_eigenvalue_after": float(min_eig_after),
            "n_training_days": split,
            "n_validation_days": n - split,
        }
        _log.info(
            "Sparse covariance: τ=%.3f, retained=%.1f%%, PSD=%s",
            tau,
            self._retained_fraction * 100,
            is_psd_after,
        )

    def _select_threshold(
        self,
        sigma_train: np.ndarray,
        sigma_val: np.ndarray,
        cfg: CovarianceConfig,
    ) -> float:
        """Grid-search for largest τ satisfying MSE ≤ max_risk_error.

        Validation MSE:
          E(τ) = (1/M) * sum_s [ x_s^T sigma_sparse(τ) x_s - x_s^T sigma_val x_s ]^2

        where x_s are M random feasible portfolios (uniform draws of K=3 stocks
        from N stocks in the validation set).
        """
        N = sigma_train.shape[0]
        rng = np.random.default_rng(0)
        M = min(cfg.n_validation_portfolios, 200)
        K_val = max(2, N // 6)  # approximate K for validation

        # Generate M random feasible portfolios
        portfolios = np.zeros((M, N), dtype=np.float64)
        for s in range(M):
            idx = rng.choice(N, size=K_val, replace=False)
            portfolios[s, idx] = 1.0

        # True portfolio risk on validation sigma
        risk_val = np.array([x @ sigma_val @ x for x in portfolios])

        best_tau = 0.0  # default: full covariance (no threshold)
        for tau in sorted(_THRESHOLD_GRID, reverse=True):
            sigma_sp = self.sparsify(sigma_train, tau)
            risk_sp = np.array([x @ sigma_sp @ x for x in portfolios])
            mse = float(np.mean((risk_sp - risk_val) ** 2))
            if mse <= cfg.max_risk_error:
                best_tau = float(tau)
                _log.debug("Auto threshold τ=%.3f accepted (MSE=%.4f)", tau, mse)
                break
            _log.debug("Auto threshold τ=%.3f rejected (MSE=%.4f > %.4f)", tau, mse, cfg.max_risk_error)

        return best_tau

    # ------------------------------------------------------------------
    # Sparsification helpers
    # ------------------------------------------------------------------

    def sparsify(self, sigma: np.ndarray, threshold: float) -> np.ndarray:
        """Apply correlation thresholding to a covariance matrix.

        Off-diagonal entries where |ρ_ij| < threshold are zeroed out.
        The diagonal is always preserved.

        Parameters
        ----------
        sigma:
            Full covariance matrix (N, N).
        threshold:
            Correlation threshold τ ∈ [0, 1).

        Returns
        -------
        np.ndarray
            Sparse covariance matrix (new array; input is not mutated).
        """
        if threshold <= 0.0:
            return sigma.copy()

        diag = np.sqrt(np.diag(sigma))
        # Avoid division by zero for zero-variance assets
        diag_safe = np.where(diag > 0, diag, 1.0)
        rho = sigma / np.outer(diag_safe, diag_safe)

        mask = np.abs(rho) >= threshold
        np.fill_diagonal(mask, True)  # always preserve diagonal

        result = sigma.copy()
        result[~mask] = 0.0
        return result

    def _compute_retained_fraction(self, sigma: np.ndarray, threshold: float) -> float:
        """Fraction of off-diagonal entries retained after thresholding."""
        diag = np.sqrt(np.diag(sigma))
        diag_safe = np.where(diag > 0, diag, 1.0)
        rho = sigma / np.outer(diag_safe, diag_safe)
        N = sigma.shape[0]
        n_offdiag = N * N - N
        if n_offdiag == 0:
            return 1.0
        # Exclude diagonal from count
        mask = np.abs(rho) >= threshold
        np.fill_diagonal(mask, False)
        return float(mask.sum()) / n_offdiag

    def check_psd(self, sigma: np.ndarray) -> tuple[bool, float]:
        """Check if a matrix is positive semidefinite.

        Returns
        -------
        (is_psd, min_eigenvalue)
            is_psd is True iff min_eigenvalue >= -1e-8.
        """
        eigvals = np.linalg.eigvalsh(sigma)
        min_eig = float(eigvals.min())
        return min_eig >= -1e-8, min_eig

    def apply_diagonal_loading(
        self,
        sigma: np.ndarray,
        lam: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Add λI to make a matrix positive semidefinite.

        Parameters
        ----------
        sigma:
            Input matrix.
        lam:
            Loading factor. If None, auto-selects lam = max(0, -min_eig) + 1e-6.

        Returns
        -------
        (sigma + lam * I, lam_used)
            Note: adds lam*K to f1 for all feasible portfolios uniformly,
            which does not change the dominance ordering within fixed K.
        """
        if lam is None:
            _, min_eig = self.check_psd(sigma)
            lam = max(0.0, -min_eig) + 1e-6
        return sigma + lam * np.eye(sigma.shape[0]), lam

    def _build_report(
        self,
        sigma: np.ndarray,
        threshold: float | None,
    ) -> dict:
        is_psd, min_eig = self.check_psd(sigma)
        return {
            "threshold": threshold,
            "retained_fraction": 1.0 if threshold is None else None,
            "psd_before_loading": is_psd,
            "psd_after_loading": is_psd,
            "diagonal_loading_applied": False,
            "lam_used": 0.0,
            "min_eigenvalue_before": float(min_eig),
            "min_eigenvalue_after": float(min_eig),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sigma_full_(self) -> np.ndarray:
        """Full annualized covariance matrix (N, N). Requires fit() to be called."""
        if self._sigma_full is None:
            raise RuntimeError("CovarianceEstimator has not been fit yet. Call .fit() first.")
        return self._sigma_full

    @property
    def sigma_sparse_(self) -> np.ndarray | None:
        """Sparse covariance matrix, or None if method='full'."""
        return self._sigma_sparse

    @property
    def threshold_(self) -> float | None:
        """Selected or specified correlation threshold."""
        return self._threshold

    @property
    def retained_fraction_(self) -> float | None:
        """Fraction of off-diagonal entries retained after thresholding."""
        return self._retained_fraction

    @property
    def sparsification_report_(self) -> dict | None:
        """Detailed sparsification metadata dict."""
        return self._report
