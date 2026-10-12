"""Financial data preprocessing: returns, annualization, diversification matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qportfolio.utils.validation import DataValidationError, validate_array_finite

TRADING_DAYS_PER_YEAR = 252


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log daily returns: ln(P_t / P_{t-1}).

    Drops the first row (which would be NaN). Raises DataValidationError
    if any NaN values remain after the calculation.

    Parameters
    ----------
    prices:
        DataFrame of adjusted close prices, DatetimeIndex, columns = tickers.

    Returns
    -------
    pd.DataFrame
        Daily log returns, same columns, len(prices)-1 rows.
    """
    if len(prices) < 2:
        raise DataValidationError(f"Need at least 2 price observations, got {len(prices)}.")

    returns = np.log(prices / prices.shift(1)).iloc[1:]

    n_nan = int(returns.isna().sum().sum())
    if n_nan > 0:
        bad = returns.columns[returns.isna().any()].tolist()
        raise DataValidationError(
            f"Daily returns contain {n_nan} NaN values in tickers: {bad}. Check price data for gaps."
        )

    return returns


def annualize_returns(
    daily_returns: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR
) -> np.ndarray:
    """Annualized expected returns: mu_i = trading_days * mean(r_i).

    Parameters
    ----------
    daily_returns:
        DataFrame of daily log returns.
    trading_days:
        Number of trading days per year (default 252).

    Returns
    -------
    np.ndarray
        Shape (N,) float64 array of annualized expected returns.
    """
    mu = np.asarray(daily_returns.mean().values) * trading_days
    validate_array_finite(mu, "annualized_returns")
    return mu.astype(np.float64)


def annualize_covariance(
    daily_returns: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR
) -> np.ndarray:
    """Annualized covariance matrix: Sigma_ij = trading_days * Cov(r_i, r_j).

    Uses Bessel's correction (ddof=1) via pandas .cov().

    Parameters
    ----------
    daily_returns:
        DataFrame of daily log returns.
    trading_days:
        Number of trading days per year (default 252).

    Returns
    -------
    np.ndarray
        Shape (N, N) float64 symmetric positive-semidefinite covariance matrix.
    """
    sigma = daily_returns.cov().values * trading_days
    validate_array_finite(sigma, "annualized_covariance")
    return sigma.astype(np.float64)


def build_diversification_matrix(tickers: list[str], sector_map: dict[str, str]) -> np.ndarray:
    """Build the sector-diversification matrix D.

    D_ij = 1 if tickers i and j are in different sectors, else 0.
    Diagonal is always 0.

    Parameters
    ----------
    tickers:
        Ordered list of ticker symbols.
    sector_map:
        Mapping {ticker: sector_name}. Missing tickers default to "Unknown".

    Returns
    -------
    np.ndarray
        Shape (N, N) float64 matrix.
    """
    N = len(tickers)
    sectors = [sector_map.get(t, "Unknown") for t in tickers]
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j and sectors[i] != sectors[j]:
                D[i, j] = 1.0
    return D


def validate_price_data(
    prices: pd.DataFrame, min_trading_days: int = 200, max_missing_fraction: float = 0.02
) -> None:
    """Raise DataValidationError if price data fails quality checks.

    Parameters
    ----------
    prices:
        Adjusted close prices DataFrame.
    min_trading_days:
        Minimum number of non-NaN rows required per ticker.
    max_missing_fraction:
        Maximum fraction of NaN values allowed per ticker.
    """
    errors = []
    total_rows = len(prices)

    for ticker in prices.columns:
        col = prices[ticker]
        n_missing = int(col.isna().sum())
        n_valid = total_rows - n_missing

        if n_valid < min_trading_days:
            errors.append(f"{ticker}: only {n_valid} valid trading days (min={min_trading_days})")
        missing_frac = n_missing / total_rows if total_rows > 0 else 1.0
        if missing_frac > max_missing_fraction:
            errors.append(
                f"{ticker}: {missing_frac:.1%} missing values (max={max_missing_fraction:.1%})"
            )
        if np.isinf(col.dropna().values).any():
            errors.append(f"{ticker}: contains infinite price values")

    if errors:
        raise DataValidationError(
            f"Price data validation failed for {len(errors)} ticker(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
