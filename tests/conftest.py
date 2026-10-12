"""Shared pytest fixtures for qportfolio tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qportfolio.problems.portfolio import PortfolioProblem

# ---------------------------------------------------------------------------
# Small synthetic problem: N=5, K=2
# ---------------------------------------------------------------------------
# Hardcoded arrays for ground-truth testing. All unit tests should use these
# to ensure determinism and reproducibility.

_TICKERS_5 = ["AAPL", "AMZN", "GOOG", "MSFT", "TSLA"]
_SECTOR_MAP_5 = {
    "AAPL": "Technology",
    "AMZN": "Consumer",
    "GOOG": "Technology",
    "MSFT": "Technology",
    "TSLA": "Consumer",
}
# Annualized expected returns
_MU_5 = np.array([0.15, 0.20, 0.12, 0.18, 0.25], dtype=np.float64)

# Annualized covariance (symmetric, PSD)
_SIGMA_5 = np.array(
    [
        [0.04, 0.02, 0.03, 0.025, 0.01],
        [0.02, 0.09, 0.01, 0.015, 0.03],
        [0.03, 0.01, 0.05, 0.04, 0.005],
        [0.025, 0.015, 0.04, 0.06, 0.008],
        [0.01, 0.03, 0.005, 0.008, 0.16],
    ],
    dtype=np.float64,
)

# D[i,j] = 1 if different sectors, 0 otherwise (diagonal = 0)
# Sectors: AAPL=Tech, AMZN=Consumer, GOOG=Tech, MSFT=Tech, TSLA=Consumer
_D_5 = np.array(
    [
        [0, 1, 0, 0, 1],  # AAPL
        [1, 0, 1, 1, 0],  # AMZN
        [0, 1, 0, 0, 1],  # GOOG
        [0, 1, 0, 0, 1],  # MSFT
        [1, 0, 1, 1, 0],  # TSLA
    ],
    dtype=np.float64,
)


@pytest.fixture
def small_problem_N5K2() -> PortfolioProblem:
    """Synthetic N=5, K=2 PortfolioProblem with hardcoded arrays. No network required."""
    return PortfolioProblem.from_arrays(
        tickers=_TICKERS_5,
        mu=_MU_5,
        sigma=_SIGMA_5,
        sector_map=_SECTOR_MAP_5,
        K=2,
        D=_D_5,
        start_date="2020-01-01",
        end_date="2022-12-31",
        seed=42,
    )


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Return a DiskCache pointing to pytest's tmp_path."""
    from qportfolio.data.cache import DiskCache

    return DiskCache(str(tmp_path / "cache"))


@pytest.fixture
def tiny_prices_df() -> pd.DataFrame:
    """5 tickers × 300 trading days of synthetic price data (deterministic)."""
    rng = np.random.default_rng(0)
    tickers = _TICKERS_5
    n_days = 300
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Geometric Brownian Motion: daily returns ~ N(0.0005, 0.015)
    log_returns = rng.normal(0.0005, 0.015, size=(n_days, len(tickers)))
    prices = np.exp(np.cumsum(log_returns, axis=0)) * 100.0

    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def mu_5() -> np.ndarray:
    return _MU_5.copy()


@pytest.fixture
def sigma_5() -> np.ndarray:
    return _SIGMA_5.copy()


@pytest.fixture
def D_5() -> np.ndarray:
    return _D_5.copy()


@pytest.fixture
def tickers_5() -> list[str]:
    return list(_TICKERS_5)


@pytest.fixture
def sector_map_5() -> dict[str, str]:
    return dict(_SECTOR_MAP_5)
