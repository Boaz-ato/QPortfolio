"""Custom exceptions and validation helpers for qportfolio."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------


class QPortfolioError(Exception):
    """Base class for all qportfolio errors."""


class DataValidationError(QPortfolioError, ValueError):
    """Raised when downloaded or provided data fails validation checks."""


class UniverseSelectionError(QPortfolioError, ValueError):
    """Raised when a stock-universe selection cannot be satisfied."""


class IntegrityError(QPortfolioError, RuntimeError):
    """Raised when a saved problem fails its problem_id integrity check."""


class InfeasibleError(QPortfolioError, ValueError):
    """Raised when a portfolio violates feasibility constraints."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_ticker_list(tickers: list[str]) -> None:
    """Raise DataValidationError on empty list or duplicate tickers."""
    if not tickers:
        raise DataValidationError("Ticker list must not be empty.")
    duplicates = {t for t in tickers if tickers.count(t) > 1}
    if duplicates:
        raise DataValidationError(f"Duplicate tickers: {sorted(duplicates)}")


def validate_array_finite(arr: np.ndarray, name: str) -> None:
    """Raise DataValidationError if array contains NaN or Inf."""
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        raise DataValidationError(
            f"Array '{name}' contains non-finite values: {n_nan} NaN, {n_inf} Inf."
        )


def validate_symmetric(arr: np.ndarray, name: str, tol: float = 1e-8) -> None:
    """Raise DataValidationError if |arr - arr.T|.max() > tol."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise DataValidationError(f"Array '{name}' must be a square 2-D matrix.")
    max_asymmetry = float(np.abs(arr - arr.T).max())
    if max_asymmetry > tol:
        raise DataValidationError(
            f"Array '{name}' is not symmetric (max |A - A^T| = {max_asymmetry:.2e})."
        )


def validate_date_range(start: str, end: str) -> None:
    """Raise ValueError if start >= end or formats are invalid (YYYY-MM-DD)."""
    from datetime import date

    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError(f"Dates must be in YYYY-MM-DD format. Got: {start!r}, {end!r}") from exc
    if s >= e:
        raise ValueError(f"start ({start}) must be strictly before end ({end}).")


def validate_shape_consistency(
    tickers: list[str], mu: np.ndarray, sigma: np.ndarray, D: np.ndarray
) -> None:
    """Raise ValueError if array shapes are inconsistent with len(tickers)."""
    N = len(tickers)
    errors = []
    if mu.shape != (N,):
        errors.append(f"mu.shape={mu.shape}, expected ({N},)")
    if sigma.shape != (N, N):
        errors.append(f"sigma.shape={sigma.shape}, expected ({N},{N})")
    if D.shape != (N, N):
        errors.append(f"D.shape={D.shape}, expected ({N},{N})")
    if errors:
        raise ValueError("Shape inconsistencies: " + "; ".join(errors))
