"""Integration tests for PortfolioProblem save/load pipeline.

These tests do NOT require network access. They use synthetic arrays to test
the full serialization pipeline including the integrity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.utils.validation import IntegrityError


@pytest.fixture
def larger_problem():
    """Synthetic N=8, K=3 problem for more thorough round-trip testing."""
    rng = np.random.default_rng(123)
    N, K = 8, 3
    tickers = [f"T{i:02d}" for i in range(N)]
    sectors = ["Tech", "Health", "Finance", "Energy"] * 2
    sector_map = {t: s for t, s in zip(tickers, sectors)}

    mu = rng.normal(0.15, 0.05, N)
    # Build a valid PSD covariance via A @ A.T
    A = rng.normal(0, 0.1, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)

    return PortfolioProblem.from_arrays(
        tickers=tickers,
        mu=mu,
        sigma=sigma,
        sector_map=sector_map,
        K=K,
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_trading_days=756,
        seed=123,
    )


def test_roundtrip_problem_id(larger_problem, tmp_path):
    save_dir = tmp_path / "p8"
    larger_problem.save(str(save_dir))
    loaded = PortfolioProblem.load(str(save_dir))
    assert larger_problem.problem_id == loaded.problem_id


def test_roundtrip_arrays_exact(larger_problem, tmp_path):
    save_dir = tmp_path / "p8_arrays"
    larger_problem.save(str(save_dir))
    loaded = PortfolioProblem.load(str(save_dir))
    np.testing.assert_array_equal(larger_problem.mu, loaded.mu)
    np.testing.assert_array_equal(larger_problem.sigma, loaded.sigma)
    np.testing.assert_array_equal(larger_problem.D, loaded.D)


def test_roundtrip_metadata(larger_problem, tmp_path):
    save_dir = tmp_path / "p8_meta"
    larger_problem.save(str(save_dir))
    loaded = PortfolioProblem.load(str(save_dir))
    assert loaded.tickers == larger_problem.tickers
    assert loaded.K == larger_problem.K
    assert loaded.sector_map == larger_problem.sector_map
    assert loaded.seed == larger_problem.seed
    assert loaded.n_trading_days == larger_problem.n_trading_days


def test_roundtrip_objectives_unchanged(larger_problem, tmp_path):
    """Evaluating objectives pre- and post-load must give identical results."""
    save_dir = tmp_path / "p8_obj"
    rng = np.random.default_rng(0)
    # Generate 5 random feasible portfolios
    portfolios = []
    for _ in range(5):
        x = np.zeros(larger_problem.N)
        idx = rng.choice(larger_problem.N, size=larger_problem.K, replace=False)
        x[idx] = 1.0
        portfolios.append(x)

    pre_load = [larger_problem.evaluate(x) for x in portfolios]

    larger_problem.save(str(save_dir))
    loaded = PortfolioProblem.load(str(save_dir))

    post_load = [loaded.evaluate(x) for x in portfolios]

    for pre, post in zip(pre_load, post_load):
        for a, b in zip(pre, post):
            assert abs(a - b) < 1e-12


def test_roundtrip_feasible_portfolios_count(tmp_path):
    """N=6, K=2: C(6,2)=15 feasible portfolios."""
    rng = np.random.default_rng(0)
    N, K = 6, 2
    tickers = [f"S{i}" for i in range(N)]
    sector_map = {t: "Tech" if i < 3 else "Finance" for i, t in enumerate(tickers)}
    A = rng.normal(0, 0.1, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)

    p = PortfolioProblem.from_arrays(
        tickers=tickers,
        mu=rng.normal(0.1, 0.05, N),
        sigma=sigma,
        sector_map=sector_map,
        K=K,
    )
    save_dir = tmp_path / "p6"
    p.save(str(save_dir))
    loaded = PortfolioProblem.load(str(save_dir))

    from math import comb

    assert len(loaded.all_feasible_portfolios()) == comb(N, K)


def test_integrity_check_on_mu_corruption(larger_problem, tmp_path):
    save_dir = tmp_path / "corrupt_mu"
    larger_problem.save(str(save_dir))

    mu_corrupt = larger_problem.mu.copy()
    mu_corrupt[0] += 1.0
    np.save(save_dir / "mu.npy", mu_corrupt)

    with pytest.raises(IntegrityError):
        PortfolioProblem.load(str(save_dir))


def test_integrity_check_on_D_corruption(larger_problem, tmp_path):
    save_dir = tmp_path / "corrupt_D"
    larger_problem.save(str(save_dir))

    D_corrupt = larger_problem.D.copy()
    D_corrupt[0, 1] = 999.0
    np.save(save_dir / "D.npy", D_corrupt)

    with pytest.raises(IntegrityError):
        PortfolioProblem.load(str(save_dir))
