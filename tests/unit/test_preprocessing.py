"""Unit tests for data preprocessing functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qportfolio.data.preprocessing import (
    annualize_covariance,
    annualize_returns,
    build_diversification_matrix,
    compute_daily_returns,
    validate_price_data,
)
from qportfolio.utils.validation import DataValidationError

# ---------------------------------------------------------------------------
# compute_daily_returns
# ---------------------------------------------------------------------------


def test_daily_returns_length(tiny_prices_df):
    returns = compute_daily_returns(tiny_prices_df)
    assert len(returns) == len(tiny_prices_df) - 1


def test_daily_returns_no_nan(tiny_prices_df):
    returns = compute_daily_returns(tiny_prices_df)
    assert not returns.isna().any().any()


def test_daily_returns_columns_preserved(tiny_prices_df):
    returns = compute_daily_returns(tiny_prices_df)
    assert list(returns.columns) == list(tiny_prices_df.columns)


def test_daily_returns_log_formula():
    """ln(2/1) = ln(2) for a single step."""
    prices = pd.DataFrame({"A": [1.0, 2.0, 4.0]})
    returns = compute_daily_returns(prices)
    expected = [np.log(2.0), np.log(2.0)]
    np.testing.assert_allclose(returns["A"].values, expected, atol=1e-12)


def test_daily_returns_too_few_rows():
    prices = pd.DataFrame({"A": [100.0]})
    with pytest.raises(DataValidationError):
        compute_daily_returns(prices)


# ---------------------------------------------------------------------------
# annualize_returns
# ---------------------------------------------------------------------------


def test_annualize_returns_known_value():
    """252 uniform daily returns of 0.001 → mu ≈ 0.252"""
    data = pd.DataFrame({"A": [0.001] * 252})
    mu = annualize_returns(data)
    assert abs(mu[0] - 0.252) < 1e-10


def test_annualize_returns_shape(tiny_prices_df):
    returns = compute_daily_returns(tiny_prices_df)
    mu = annualize_returns(returns)
    assert mu.shape == (len(tiny_prices_df.columns),)
    assert mu.dtype == np.float64


def test_annualize_returns_custom_trading_days():
    data = pd.DataFrame({"A": [0.001] * 100})
    mu = annualize_returns(data, trading_days=100)
    assert abs(mu[0] - 0.1) < 1e-10


# ---------------------------------------------------------------------------
# annualize_covariance
# ---------------------------------------------------------------------------


def test_annualize_covariance_symmetric(tiny_prices_df):
    returns = compute_daily_returns(tiny_prices_df)
    sigma = annualize_covariance(returns)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)


def test_annualize_covariance_psd(tiny_prices_df):
    returns = compute_daily_returns(tiny_prices_df)
    sigma = annualize_covariance(returns)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() >= -1e-8, f"min eigenvalue = {eigvals.min()}"


def test_annualize_covariance_diagonal_known():
    """Single asset, known variance: Sigma_00 = 252 * Var(r, ddof=1)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 300)
    data = pd.DataFrame({"A": r})
    sigma = annualize_covariance(data)
    expected = 252 * np.var(r, ddof=1)
    assert abs(sigma[0, 0] - expected) < 1e-10


def test_annualize_covariance_shape(tiny_prices_df):
    returns = compute_daily_returns(tiny_prices_df)
    sigma = annualize_covariance(returns)
    N = len(tiny_prices_df.columns)
    assert sigma.shape == (N, N)


# ---------------------------------------------------------------------------
# build_diversification_matrix
# ---------------------------------------------------------------------------


def test_diversification_matrix_structure():
    """4 tickers in 2 sectors (2+2). Verify exact matrix."""
    tickers = ["A", "B", "C", "D"]
    sector_map = {"A": "Tech", "B": "Tech", "C": "Finance", "D": "Finance"}
    D = build_diversification_matrix(tickers, sector_map)

    # Same sector: D=0; different sector: D=1; diagonal: D=0
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(D, expected)


def test_diversification_matrix_all_same_sector():
    """All tickers in one sector → D is all zeros."""
    tickers = ["A", "B", "C"]
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech"}
    D = build_diversification_matrix(tickers, sector_map)
    np.testing.assert_array_equal(D, np.zeros((3, 3)))


def test_diversification_matrix_diagonal_zero():
    """Diagonal must always be 0."""
    tickers = ["A", "B", "C", "D", "E"]
    sector_map = {t: f"Sector{i}" for i, t in enumerate(tickers)}
    D = build_diversification_matrix(tickers, sector_map)
    np.testing.assert_array_equal(np.diag(D), np.zeros(5))


def test_diversification_matrix_symmetric():
    tickers = ["A", "B", "C"]
    sector_map = {"A": "Tech", "B": "Health", "C": "Tech"}
    D = build_diversification_matrix(tickers, sector_map)
    np.testing.assert_array_equal(D, D.T)


def test_diversification_matrix_missing_ticker():
    """Tickers missing from sector_map default to 'Unknown'."""
    tickers = ["A", "B"]
    sector_map = {"A": "Tech"}  # B is missing
    D = build_diversification_matrix(tickers, sector_map)
    # A=Tech, B=Unknown → different sectors → D[0,1]=D[1,0]=1
    assert D[0, 1] == 1.0
    assert D[1, 0] == 1.0


# ---------------------------------------------------------------------------
# validate_price_data
# ---------------------------------------------------------------------------


def test_validate_price_data_passes(tiny_prices_df):
    validate_price_data(tiny_prices_df, min_trading_days=200, max_missing_fraction=0.02)


def test_validate_price_data_fails_on_too_few_days():
    prices = pd.DataFrame({"A": [100.0] * 50, "B": [200.0] * 50})
    with pytest.raises(DataValidationError, match="valid trading days"):
        validate_price_data(prices, min_trading_days=100)


def test_validate_price_data_fails_on_too_many_nans():
    prices = pd.DataFrame({"A": [100.0] * 200 + [float("nan")] * 10})
    with pytest.raises(DataValidationError, match="missing"):
        validate_price_data(prices, min_trading_days=50, max_missing_fraction=0.01)
