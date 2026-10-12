"""Unit tests for ConstraintValidator."""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.problems.constraints import ConstraintValidator


@pytest.fixture
def cv():
    return ConstraintValidator(N=5, K=2)


def test_is_feasible_correct(cv):
    x = np.array([1, 1, 0, 0, 0], dtype=float)
    assert cv.is_feasible(x) is True


def test_is_feasible_wrong_cardinality(cv):
    x = np.array([1, 0, 0, 0, 0], dtype=float)  # K=1, not 2
    assert cv.is_feasible(x) is False


def test_is_feasible_non_binary(cv):
    x = np.array([0.5, 1.5, 0, 0, 0], dtype=float)
    assert cv.is_feasible(x) is False


def test_is_feasible_wrong_length(cv):
    x = np.array([1, 1, 0], dtype=float)
    assert cv.is_feasible(x) is False


def test_is_feasible_batch(cv):
    X = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0],  # K=1 — infeasible
            [0.5, 1, 0, 0, 0],  # non-binary — infeasible
        ],
        dtype=float,
    )
    result = cv.is_feasible_batch(X)
    assert list(result) == [True, True, False, False]


def test_feasibility_report_feasible(cv):
    x = np.array([1, 0, 1, 0, 0], dtype=float)
    r = cv.feasibility_report(x)
    assert r["is_feasible"] is True
    assert r["cardinality"] == 2
    assert r["violations"] == []


def test_feasibility_report_wrong_k(cv):
    x = np.array([1, 1, 1, 0, 0], dtype=float)  # K=3
    r = cv.feasibility_report(x)
    assert r["is_feasible"] is False
    assert any("cardinality" in v for v in r["violations"])


def test_feasibility_report_non_binary(cv):
    x = np.array([0.3, 1, 0, 0, 0], dtype=float)
    r = cv.feasibility_report(x)
    assert r["is_feasible"] is False


def test_all_feasible_portfolios_N4K2():
    """C(4,2)=6; all rows must sum to 2 and be binary."""
    cv = ConstraintValidator(N=4, K=2)
    portfolios = cv.all_feasible_portfolios()
    assert portfolios.shape == (6, 4)
    assert (portfolios.sum(axis=1) == 2).all()
    assert set(np.unique(portfolios)).issubset({0, 1})


def test_all_feasible_portfolios_unique():
    cv = ConstraintValidator(N=5, K=2)
    portfolios = cv.all_feasible_portfolios()
    # All rows unique
    assert len(set(map(tuple, portfolios.tolist()))) == len(portfolios)


def test_all_feasible_portfolios_count():
    from math import comb

    for N, K in [(4, 2), (5, 2), (6, 3), (8, 4)]:
        cv = ConstraintValidator(N=N, K=K)
        expected = comb(N, K)
        portfolios = cv.all_feasible_portfolios()
        assert len(portfolios) == expected, f"C({N},{K}): expected {expected}, got {len(portfolios)}"


def test_all_feasible_portfolios_overflow_guard():
    cv = ConstraintValidator(N=30, K=15)  # C(30,15) = 155_117_520 > 1_000_000
    with pytest.raises(OverflowError, match="safety limit"):
        cv.all_feasible_portfolios()


def test_constructor_invalid_K():
    with pytest.raises(ValueError):
        ConstraintValidator(N=5, K=0)
    with pytest.raises(ValueError):
        ConstraintValidator(N=5, K=6)  # K > N


def test_constructor_invalid_N():
    with pytest.raises(ValueError):
        ConstraintValidator(N=0, K=1)
