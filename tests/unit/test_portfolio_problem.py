"""Unit tests for PortfolioProblem."""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.utils.validation import IntegrityError

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_from_arrays(small_problem_N5K2):
    p = small_problem_N5K2
    assert p.N == 5
    assert p.K == 2
    assert len(p.tickers) == 5
    assert p.problem_id != ""
    assert len(p.problem_id) == 16


def test_construct_validates_shapes(mu_5, sigma_5, D_5, tickers_5, sector_map_5):
    with pytest.raises(ValueError, match="Shape inconsistencies"):
        PortfolioProblem.from_arrays(
            tickers=tickers_5,
            mu=mu_5[:3],  # wrong shape
            sigma=sigma_5,
            sector_map=sector_map_5,
            K=2,
        )


def test_construct_validates_duplicate_tickers(mu_5, sigma_5, sector_map_5):
    with pytest.raises(Exception):  # DataValidationError
        PortfolioProblem.from_arrays(
            tickers=["A", "A", "B", "C", "D"],
            mu=mu_5,
            sigma=sigma_5,
            sector_map={"A": "Tech", "B": "Health", "C": "Finance", "D": "Energy"},
            K=2,
        )


# ---------------------------------------------------------------------------
# problem_id
# ---------------------------------------------------------------------------


def test_problem_id_deterministic(small_problem_N5K2, mu_5, sigma_5, D_5, sector_map_5, tickers_5):
    p2 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_5, sigma=sigma_5, sector_map=sector_map_5, K=2, D=D_5
    )
    assert small_problem_N5K2.problem_id == p2.problem_id


def test_problem_id_invariant_to_ticker_order(mu_5, sigma_5, D_5, sector_map_5, tickers_5):
    """Shuffling tickers (with consistent array permutation) must yield same id."""
    perm = [4, 2, 0, 3, 1]  # arbitrary permutation
    tickers_perm = [tickers_5[i] for i in perm]
    mu_perm = mu_5[perm]
    sigma_perm = sigma_5[np.ix_(perm, perm)]
    D_perm = D_5[np.ix_(perm, perm)]
    sector_map_perm = {tickers_perm[i]: sector_map_5[tickers_5[perm[i]]] for i in range(5)}

    p1 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_5, sigma=sigma_5, sector_map=sector_map_5, K=2, D=D_5
    )
    p2 = PortfolioProblem.from_arrays(
        tickers=tickers_perm, mu=mu_perm, sigma=sigma_perm, sector_map=sector_map_perm, K=2, D=D_perm
    )
    assert p1.problem_id == p2.problem_id


def test_problem_id_changes_on_mu_change(mu_5, sigma_5, D_5, sector_map_5, tickers_5):
    p1 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_5, sigma=sigma_5, sector_map=sector_map_5, K=2, D=D_5
    )
    mu_perturbed = mu_5.copy()
    mu_perturbed[0] += 1e-6  # well above 8-digit rounding precision (5e-9)
    p2 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_perturbed, sigma=sigma_5, sector_map=sector_map_5, K=2, D=D_5
    )
    assert p1.problem_id != p2.problem_id


def test_problem_id_changes_on_K_change(mu_5, sigma_5, D_5, sector_map_5, tickers_5):
    p1 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_5, sigma=sigma_5, sector_map=sector_map_5, K=2, D=D_5
    )
    p2 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_5, sigma=sigma_5, sector_map=sector_map_5, K=3, D=D_5
    )
    assert p1.problem_id != p2.problem_id


def test_problem_id_unchanged_by_sector_map(mu_5, sigma_5, D_5, sector_map_5, tickers_5):
    """Sector map is NOT part of the problem_id (it's metadata, not the optimization problem)."""
    p1 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_5, sigma=sigma_5, sector_map=sector_map_5, K=2, D=D_5
    )
    different_sectors = {t: "Unknown" for t in tickers_5}
    p2 = PortfolioProblem.from_arrays(
        tickers=tickers_5, mu=mu_5, sigma=sigma_5, sector_map=different_sectors, K=2, D=D_5
    )
    assert p1.problem_id == p2.problem_id


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(small_problem_N5K2, tmp_path):
    p = small_problem_N5K2
    save_dir = tmp_path / "problem"
    p.save(str(save_dir))

    # Check files were created
    assert (save_dir / "problem.json").exists()
    assert (save_dir / "mu.npy").exists()
    assert (save_dir / "sigma.npy").exists()
    assert (save_dir / "D.npy").exists()

    p2 = PortfolioProblem.load(str(save_dir))

    assert p.problem_id == p2.problem_id
    assert p.tickers == p2.tickers
    assert p.K == p2.K
    np.testing.assert_array_almost_equal(p.mu, p2.mu, decimal=12)
    np.testing.assert_array_almost_equal(p.sigma, p2.sigma, decimal=12)
    np.testing.assert_array_almost_equal(p.D, p2.D, decimal=12)


def test_save_load_sparse_covariance(small_problem_N5K2, tmp_path):
    """sigma_sparse is saved and loaded when present."""
    p = small_problem_N5K2
    # Manually attach a sparse sigma
    sigma_sparse = p.sigma.copy()
    sigma_sparse[0, 1] = sigma_sparse[1, 0] = 0.0
    import dataclasses

    p2 = dataclasses.replace(p, sigma_sparse=sigma_sparse)

    save_dir = tmp_path / "problem_sparse"
    p2.save(str(save_dir))
    assert (save_dir / "sigma_sparse.npy").exists()

    p3 = PortfolioProblem.load(str(save_dir))
    assert p3.sigma_sparse is not None
    np.testing.assert_array_almost_equal(p3.sigma_sparse, sigma_sparse)


def test_load_integrity_check_fails_on_corruption(small_problem_N5K2, tmp_path):
    p = small_problem_N5K2
    save_dir = tmp_path / "problem_corrupt"
    p.save(str(save_dir))

    # Corrupt the sigma array
    sigma_corrupt = p.sigma.copy()
    sigma_corrupt[0, 0] += 99.0
    np.save(save_dir / "sigma.npy", sigma_corrupt)

    with pytest.raises(IntegrityError, match="problem_id mismatch"):
        PortfolioProblem.load(str(save_dir))


# ---------------------------------------------------------------------------
# Objective delegation
# ---------------------------------------------------------------------------


def test_evaluate_delegation(small_problem_N5K2):
    x = np.array([1, 0, 1, 0, 0], dtype=float)
    from qportfolio.problems.objectives import ObjectiveEvaluator

    direct = ObjectiveEvaluator(
        small_problem_N5K2.mu, small_problem_N5K2.sigma, small_problem_N5K2.D
    ).evaluate(x)
    via_problem = small_problem_N5K2.evaluate(x)
    assert direct == via_problem


def test_evaluate_batch_delegation(small_problem_N5K2):
    X = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
        ],
        dtype=float,
    )
    result = small_problem_N5K2.evaluate_batch(X)
    assert result.shape == (2, 3)


def test_all_feasible_portfolios_via_problem(small_problem_N5K2):
    from math import comb

    portfolios = small_problem_N5K2.all_feasible_portfolios()
    assert len(portfolios) == comb(5, 2)  # 10


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_has_required_keys(small_problem_N5K2):
    s = small_problem_N5K2.summary()
    for key in ["problem_id", "N", "K", "tickers", "date_range"]:
        assert key in s, f"Missing key: {key}"
