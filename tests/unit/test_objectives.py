"""Unit tests for ObjectiveEvaluator.

Critical ground-truth tests: enumerate all C(5,2)=10 portfolios by hand
and verify every objective triple to 1e-10.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qportfolio.problems.objectives import ObjectiveEvaluator

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

# N=5 hardcoded arrays (same as conftest)
_TICKERS = ["AAPL", "AMZN", "GOOG", "MSFT", "TSLA"]
_MU = np.array([0.15, 0.20, 0.12, 0.18, 0.25], dtype=np.float64)
_SIGMA = np.array(
    [
        [0.04, 0.02, 0.03, 0.025, 0.01],
        [0.02, 0.09, 0.01, 0.015, 0.03],
        [0.03, 0.01, 0.05, 0.04, 0.005],
        [0.025, 0.015, 0.04, 0.06, 0.008],
        [0.01, 0.03, 0.005, 0.008, 0.16],
    ],
    dtype=np.float64,
)
_D = np.array(
    [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 0],
    ],
    dtype=np.float64,
)


@pytest.fixture
def evaluator():
    return ObjectiveEvaluator(mu=_MU, sigma=_SIGMA, D=_D)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_construct_evaluator(evaluator):
    assert evaluator.N == 5


def test_construct_rejects_non_square_sigma():
    with pytest.raises(ValueError):
        ObjectiveEvaluator(mu=_MU, sigma=np.ones((5, 4)), D=_D)


def test_construct_rejects_asymmetric_sigma():
    sigma_bad = _SIGMA.copy()
    sigma_bad[0, 1] = 999.0
    with pytest.raises(Exception):  # DataValidationError or ValueError
        ObjectiveEvaluator(mu=_MU, sigma=sigma_bad, D=_D)


def test_construct_rejects_nan_mu():
    mu_bad = _MU.copy()
    mu_bad[0] = float("nan")
    with pytest.raises(Exception):
        ObjectiveEvaluator(mu=mu_bad, sigma=_SIGMA, D=_D)


# ---------------------------------------------------------------------------
# f2: expected return (simplest, exact formula)
# ---------------------------------------------------------------------------


def test_f2_known_return(evaluator):
    """x=[0,1,1,0,0]: f2 = mu[1] + mu[2] = 0.20 + 0.12 = 0.32"""
    x = np.array([0, 1, 1, 0, 0], dtype=float)
    _, f2, _ = evaluator.evaluate(x)
    assert abs(f2 - 0.32) < 1e-12


def test_f2_first_and_last(evaluator):
    """x=[1,0,0,0,1]: f2 = mu[0] + mu[4] = 0.15 + 0.25 = 0.40"""
    x = np.array([1, 0, 0, 0, 1], dtype=float)
    _, f2, _ = evaluator.evaluate(x)
    assert abs(f2 - 0.40) < 1e-12


# ---------------------------------------------------------------------------
# f1: negative risk — spot checks against manual computation
# ---------------------------------------------------------------------------


def test_f1_known_portfolio(evaluator):
    """x=[1,1,0,0,0]: f1 = -(sigma_00 + sigma_11 + 2*sigma_01)
    = -(0.04 + 0.09 + 2*0.02) = -0.17
    """
    x = np.array([1, 1, 0, 0, 0], dtype=float)
    f1, _, _ = evaluator.evaluate(x)
    expected = -(0.04 + 0.09 + 2 * 0.02)
    assert abs(f1 - expected) < 1e-12


def test_f1_diagonal_only(evaluator):
    """x=[1,0,0,0,1]: f1 = -(sigma_00 + sigma_44 + 2*sigma_04)
    = -(0.04 + 0.16 + 2*0.01) = -0.22
    """
    x = np.array([1, 0, 0, 0, 1], dtype=float)
    f1, _, _ = evaluator.evaluate(x)
    expected = -(0.04 + 0.16 + 2 * 0.01)
    assert abs(f1 - expected) < 1e-12


# ---------------------------------------------------------------------------
# f3: diversification
# ---------------------------------------------------------------------------


def test_f3_same_sector_is_zero(evaluator):
    """AAPL(Tech) + GOOG(Tech): f3 = 0 (D[0,2] = D[2,0] = 0)"""
    x = np.array([1, 0, 1, 0, 0], dtype=float)
    _, _, f3 = evaluator.evaluate(x)
    assert abs(f3) < 1e-12


def test_f3_different_sectors_is_two(evaluator):
    """AAPL(Tech) + AMZN(Consumer): D[0,1]=D[1,0]=1, diagonal=0
    f3 = x^T D x = 2*D[0,1]*x[0]*x[1] = 2
    """
    x = np.array([1, 1, 0, 0, 0], dtype=float)
    _, _, f3 = evaluator.evaluate(x)
    assert abs(f3 - 2.0) < 1e-12


def test_f3_known_3stock(evaluator):
    """AAPL(Tech) + AMZN(Consumer): 2 cross-sector pairs → f3=2 (see above)."""
    x = np.array([1, 1, 0, 0, 0], dtype=float)
    _, _, f3 = evaluator.evaluate(x)
    assert abs(f3 - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# GROUND TRUTH: enumerate all C(5,2)=10 portfolios
# ---------------------------------------------------------------------------


def _reference_objectives(mu, sigma, D):
    """Compute all 10 portfolios' objectives using a naive (reference) implementation."""
    results = {}
    for combo in itertools.combinations(range(5), 2):
        x = np.zeros(5)
        x[list(combo)] = 1.0
        f1 = float(-x @ sigma @ x)
        f2 = float(mu @ x)
        f3 = float(x @ D @ x)
        results[combo] = (f1, f2, f3)
    return results


def test_evaluate_all_feasible_N5K2(evaluator):
    """Enumerate all 10 K=2 portfolios; verify library matches reference to 1e-10."""
    ref = _reference_objectives(_MU, _SIGMA, _D)

    for combo, (ref_f1, ref_f2, ref_f3) in ref.items():
        x = np.zeros(5)
        x[list(combo)] = 1.0
        f1, f2, f3 = evaluator.evaluate(x)
        assert abs(f1 - ref_f1) < 1e-10, f"f1 mismatch for combo {combo}: {f1} vs {ref_f1}"
        assert abs(f2 - ref_f2) < 1e-10, f"f2 mismatch for combo {combo}: {f2} vs {ref_f2}"
        assert abs(f3 - ref_f3) < 1e-10, f"f3 mismatch for combo {combo}: {f3} vs {ref_f3}"


def test_evaluate_batch_matches_single(evaluator):
    """evaluate_batch(X) must match evaluate(x) for every row, within 1e-12."""
    # Generate all 10 K=2 portfolios
    combos = list(itertools.combinations(range(5), 2))
    X = np.zeros((len(combos), 5))
    for i, combo in enumerate(combos):
        X[i, list(combo)] = 1.0

    batch = evaluator.evaluate_batch(X)

    for i, combo in enumerate(combos):
        single = evaluator.evaluate(X[i])
        for j in range(3):
            assert abs(batch[i, j] - single[j]) < 1e-12, (
                f"Objective {j} mismatch at combo {combo}: batch={batch[i, j]}, single={single[j]}"
            )


def test_evaluate_batch_shape(evaluator):
    """evaluate_batch returns (M, 3) for M portfolios."""
    X = np.zeros((7, 5))
    for i in range(7):
        idx = np.random.default_rng(i).choice(5, size=2, replace=False)
        X[i, idx] = 1.0
    out = evaluator.evaluate_batch(X)
    assert out.shape == (7, 3)


def test_evaluate_batch_single_row(evaluator):
    """evaluate_batch handles a single-row (1,N) input."""
    X = np.array([[1, 0, 0, 1, 0]], dtype=float)
    out = evaluator.evaluate_batch(X)
    single = evaluator.evaluate(X[0])
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], single, atol=1e-12)


# ---------------------------------------------------------------------------
# Sign and formula checks
# ---------------------------------------------------------------------------


def test_f1_is_negative_or_zero(evaluator):
    """f1 = -x^T Sigma x should always be ≤ 0 for PSD sigma and non-zero x."""
    rng = np.random.default_rng(99)
    for _ in range(20):
        idx = rng.choice(5, size=2, replace=False)
        x = np.zeros(5)
        x[idx] = 1.0
        f1, _, _ = evaluator.evaluate(x)
        assert f1 <= 1e-10, f"f1={f1} > 0 for x={x}"


def test_f3_is_non_negative(evaluator):
    """f3 = x^T D x should always be >= 0 since D has non-negative entries."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        idx = rng.choice(5, size=2, replace=False)
        x = np.zeros(5)
        x[idx] = 1.0
        _, _, f3 = evaluator.evaluate(x)
        assert f3 >= -1e-12, f"f3={f3} < 0 for x={x}"


def test_objective_names(evaluator):
    assert evaluator.objective_names() == ["negative_risk", "expected_return", "diversification"]


def test_evaluate_individual(evaluator):
    """evaluate_individual(x, j) matches evaluate(x)[j]."""
    x = np.array([1, 0, 0, 1, 0], dtype=float)
    f1, f2, f3 = evaluator.evaluate(x)
    assert evaluator.evaluate_individual(x, 0) == f1
    assert evaluator.evaluate_individual(x, 1) == f2
    assert evaluator.evaluate_individual(x, 2) == f3


def test_evaluate_individual_invalid_index(evaluator):
    x = np.zeros(5)
    with pytest.raises(ValueError):
        evaluator.evaluate_individual(x, 3)
