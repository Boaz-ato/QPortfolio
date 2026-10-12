"""Unit tests for the Phase 3 evaluation engine.

Covers:
  - SolverTimer
  - SolutionDiscoveryEvent / QAOATiming
  - NormalizedHypervolume
  - compute_hypervolume
  - compute_reference_point (strategies: exact, conservative, manual)
  - build_normalizer
  - BenchmarkSuite / ComparisonTable
  - PortfolioProblem.compare()
"""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.evaluation.benchmark import (
    BenchmarkMode,
    BenchmarkSuite,
    ComparisonTable,
)
from qportfolio.evaluation.hypervolume import (
    NormalizedHypervolume,
    _pymoo_hv,
    compute_hypervolume,
)
from qportfolio.evaluation.reference import (
    build_normalizer,
    compute_reference_point,
)
from qportfolio.evaluation.timing import QAOATiming, SolutionDiscoveryEvent, SolverTimer
from qportfolio.problems.portfolio import PortfolioProblem

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def problem_N8K4():
    rng = np.random.default_rng(7)
    N, K = 8, 4
    tickers = [f"S{i:02d}" for i in range(N)]
    sectors = ["Tech", "Health", "Finance", "Energy"] * 2
    sector_map = {t: s for t, s in zip(tickers, sectors)}
    mu = rng.uniform(0.08, 0.28, N)
    A = rng.normal(0, 0.06, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)
    return PortfolioProblem.from_arrays(
        tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K, seed=7
    )


# ---------------------------------------------------------------------------
# SolverTimer
# ---------------------------------------------------------------------------


def test_solver_timer_starts_at_zero():
    timer = SolverTimer()
    assert timer.elapsed() == 0.0


def test_solver_timer_measures_elapsed():
    import time

    timer = SolverTimer()
    timer.start()
    time.sleep(0.02)
    assert timer.elapsed() >= 0.01


def test_solver_timer_stop_returns_elapsed():
    import time

    timer = SolverTimer()
    timer.start()
    time.sleep(0.01)
    t = timer.stop()
    assert t >= 0.01
    # After stop, elapsed() returns 0
    assert timer.elapsed() == 0.0


# ---------------------------------------------------------------------------
# SolutionDiscoveryEvent
# ---------------------------------------------------------------------------


def test_solution_discovery_event_fields():
    evt = SolutionDiscoveryEvent(
        elapsed_time=1.23,
        candidate_id="abc",
        objective_values=(0.1, 0.2, 0.3),
        feasible=True,
        n_nondominated=5,
        hypervolume=0.42,
        source="test",
    )
    assert evt.elapsed_time == 1.23
    assert evt.feasible is True
    assert evt.n_nondominated == 5


# ---------------------------------------------------------------------------
# QAOATiming
# ---------------------------------------------------------------------------


def test_qaoa_timing_t_cold():
    t = QAOATiming(training_time=10.0, sampling_time=2.0, other_time=0.5)
    assert abs(t.t_cold - 12.5) < 1e-9


def test_qaoa_timing_t_amortized():
    t = QAOATiming(training_time=10.0, sampling_time=2.0, other_time=0.5, n_repetitions=5)
    assert abs(t.t_amortized - (10.0 / 5 + 2.0 + 0.5)) < 1e-9


def test_qaoa_timing_to_dict():
    t = QAOATiming(training_time=1.0, sampling_time=2.0, other_time=0.0)
    d = t.to_dict()
    assert "t_cold" in d
    assert "t_amortized" in d


# ---------------------------------------------------------------------------
# NormalizedHypervolume
# ---------------------------------------------------------------------------


def test_normalizer_scale_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        NormalizedHypervolume(shift=np.zeros(3), scale=np.array([1.0, 0.0, 1.0]))


def test_normalizer_normalize_identity():
    """With shift=0, scale=1 the normalization is the identity."""
    norm = NormalizedHypervolume(shift=np.zeros(3), scale=np.ones(3))
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(norm.normalize(pts), pts)


def test_normalizer_normalize_correct():
    norm = NormalizedHypervolume(shift=np.array([1.0, 0.0, -1.0]), scale=np.array([2.0, 4.0, 0.5]))
    pt = np.array([3.0, 8.0, -0.5])
    expected = np.array([(3 - 1) / 2, (8 - 0) / 4, (-0.5 - (-1)) / 0.5])
    np.testing.assert_allclose(norm.normalize(pt), expected)


def test_normalizer_from_reference_range():
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([1.0, 3.0, 4.0])
    norm = NormalizedHypervolume.from_reference_range(lo, hi)
    # scale = hi - lo = [1, 2, 2]
    np.testing.assert_allclose(norm.scale, [1.0, 2.0, 2.0])
    np.testing.assert_allclose(norm.shift, lo)


def test_normalizer_from_reference_range_zero_span():
    """Zero-span objectives get scale=1 instead of 0."""
    lo = np.array([1.0, 1.0, 1.0])
    hi = np.array([1.0, 1.0, 2.0])
    norm = NormalizedHypervolume.from_reference_range(lo, hi)
    assert norm.scale[0] == 1.0  # zero-span guarded
    assert norm.scale[2] == 1.0  # non-zero span preserved (= 1.0)


# ---------------------------------------------------------------------------
# Hypervolume computation
# ---------------------------------------------------------------------------


def test_pymoo_hv_empty_returns_zero():
    ref = np.array([0.0, 0.0, 0.0])
    assert _pymoo_hv(np.zeros((0, 3)), ref) == 0.0


def test_pymoo_hv_single_point():
    """A single point above the reference point should give positive HV."""
    ref = np.array([0.0, 0.0, 0.0])
    pts = np.array([[1.0, 1.0, 1.0]])
    hv = _pymoo_hv(pts, ref)
    assert hv == pytest.approx(1.0)


def test_compute_hypervolume_raw():
    ref = np.array([0.0, 0.0])
    pts = np.array([[2.0, 1.0], [1.0, 2.0]])
    hv = compute_hypervolume(pts, ref)
    assert hv > 0.0


def test_compute_hypervolume_with_normalizer():
    shift = np.array([0.0, 0.0])
    scale = np.array([2.0, 2.0])
    norm = NormalizedHypervolume(shift=shift, scale=scale)
    ref = np.array([0.0, 0.0])
    pts = np.array([[2.0, 0.0], [0.0, 2.0]])
    hv_raw = compute_hypervolume(pts, ref)
    hv_norm = compute_hypervolume(pts, ref, normalizer=norm)
    # Normalized HV = raw HV / scale^D
    assert hv_norm == pytest.approx(hv_raw / (2.0 * 2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# Reference point computation
# ---------------------------------------------------------------------------


def test_reference_point_manual():
    """manual strategy returns the given point unchanged."""
    manual = np.array([-99.0, -99.0, -99.0])
    problem = None  # not accessed for manual strategy
    ref = compute_reference_point(problem, strategy="manual", manual_point=manual)
    np.testing.assert_array_equal(ref, manual)


def test_reference_point_manual_requires_point():
    with pytest.raises(ValueError, match="manual_point"):
        compute_reference_point(None, strategy="manual")


def test_reference_point_exact_dominated(problem_N8K4):
    """Exact reference point must be strictly dominated by every feasible portfolio."""
    ref = compute_reference_point(problem_N8K4, strategy="exact")
    all_x = problem_N8K4.all_feasible_portfolios()
    all_objs = problem_N8K4.evaluate_batch(all_x.astype(float))
    # Every objective of every portfolio must exceed the reference point
    assert np.all(all_objs > ref), f"Reference point {ref} not strictly dominated by all portfolios"


def test_reference_point_conservative_dominated(problem_N8K4):
    """Conservative reference point must also be dominated by all feasible portfolios."""
    ref = compute_reference_point(problem_N8K4, strategy="conservative")
    all_x = problem_N8K4.all_feasible_portfolios()
    all_objs = problem_N8K4.evaluate_batch(all_x.astype(float))
    assert np.all(all_objs > ref), f"Conservative reference point {ref} not dominated"


def test_reference_point_exact_vs_conservative(problem_N8K4):
    """Exact reference point should be tighter (closer) than conservative."""
    ref_exact = compute_reference_point(problem_N8K4, strategy="exact")
    ref_cons = compute_reference_point(problem_N8K4, strategy="conservative")
    # exact should be >= conservative (or equal) in each dimension
    assert np.all(ref_exact >= ref_cons - 1e-3), (
        "Exact reference point should not be strictly looser than conservative"
    )


def test_reference_point_shape(problem_N8K4):
    ref = compute_reference_point(problem_N8K4, strategy="exact")
    assert ref.shape == (3,)


# ---------------------------------------------------------------------------
# build_normalizer
# ---------------------------------------------------------------------------


def test_build_normalizer_exact(problem_N8K4):
    norm = build_normalizer(problem_N8K4, strategy="exact")
    assert isinstance(norm, NormalizedHypervolume)
    assert norm.scale.shape == (3,)
    assert np.all(norm.scale > 0)


def test_build_normalizer_normalized_range(problem_N8K4):
    """After normalization, the Pareto front should lie in roughly [0, 1]."""
    norm = build_normalizer(problem_N8K4, strategy="exact")
    all_x = problem_N8K4.all_feasible_portfolios()
    all_objs = problem_N8K4.evaluate_batch(all_x.astype(float))
    norm_objs = norm.normalize(all_objs)
    # All normalized values should be in [0, 1] (exact bounds → exact range)
    assert np.all(norm_objs >= -1e-9)
    assert np.all(norm_objs <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


def test_benchmark_suite_runs(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler
    from qportfolio.solvers.wsm import WSMSolver

    suite = BenchmarkSuite(
        problem_N8K4,
        solvers=[
            RandomSampler(n_samples=200, seed=0),
            WSMSolver(n_weights=5, seed=0),
        ],
        mode=BenchmarkMode.FIXED_TIME,
        budget=10.0,
    )
    results = suite.run()
    assert len(results) == 2
    for r in results:
        assert r.n_feasible >= 0
        assert r.runtime >= 0.0


def test_benchmark_suite_comparison_table(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler

    suite = BenchmarkSuite(
        problem_N8K4,
        solvers=[RandomSampler(n_samples=300, seed=0)],
        mode=BenchmarkMode.FIXED_TIME,
        budget=10.0,
    )
    table = suite.comparison_table()
    assert isinstance(table, ComparisonTable)
    assert len(table.rows) == 1
    assert "algorithm" in table.rows[0]
    assert "hypervolume" in table.rows[0]


def test_benchmark_suite_hypervolume_nonneg(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler

    suite = BenchmarkSuite(
        problem_N8K4,
        solvers=[RandomSampler(n_samples=500, seed=0)],
        mode=BenchmarkMode.FIXED_TIME,
        budget=10.0,
    )
    results = suite.run()
    for r in results:
        if r.hypervolume is not None:
            assert r.hypervolume >= 0.0


def test_benchmark_suite_custom_reference_point(problem_N8K4):
    """Suite should respect a user-provided reference point."""
    from qportfolio.solvers.random import RandomSampler

    ref = compute_reference_point(problem_N8K4, strategy="exact")
    suite = BenchmarkSuite(
        problem_N8K4,
        solvers=[RandomSampler(n_samples=200, seed=0)],
        reference_point=ref,
    )
    suite.run()
    np.testing.assert_array_equal(suite._reference_point, ref)


# ---------------------------------------------------------------------------
# ComparisonTable.to_string
# ---------------------------------------------------------------------------


def test_comparison_table_to_string():
    table = ComparisonTable(
        problem_id="abc123",
        mode="comparison",
        budget=float("nan"),
        reference_point=np.array([0.0, 0.0, 0.0]),
        rows=[
            {"algorithm": "random", "n_nondominated": 3, "hypervolume": 0.5},
            {"algorithm": "wsm", "n_nondominated": 5, "hypervolume": None},
        ],
    )
    s = table.to_string()
    assert "random" in s
    assert "wsm" in s
    assert "N/A" in s


# ---------------------------------------------------------------------------
# PortfolioProblem.compare()
# ---------------------------------------------------------------------------


def test_problem_compare_returns_table(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler
    from qportfolio.solvers.wsm import WSMSolver

    r1 = RandomSampler(n_samples=500, seed=0).solve(problem_N8K4)
    r2 = WSMSolver(n_weights=10, seed=0).solve(problem_N8K4)

    table = problem_N8K4.compare([r1, r2])
    assert isinstance(table, ComparisonTable)
    assert len(table.rows) == 2
    algos = {row["algorithm"] for row in table.rows}
    assert "random" in algos
    assert "wsm" in algos


def test_problem_compare_problem_id(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler

    result = RandomSampler(n_samples=200, seed=0).solve(problem_N8K4)
    table = problem_N8K4.compare([result])
    assert table.problem_id == problem_N8K4.problem_id


def test_problem_compare_hv_nonneg(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler

    result = RandomSampler(n_samples=1000, seed=0).solve(problem_N8K4)
    table = problem_N8K4.compare([result])
    for row in table.rows:
        if row["hypervolume"] is not None:
            assert row["hypervolume"] >= 0.0


def test_problem_compare_manual_reference(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler

    ref = compute_reference_point(problem_N8K4, strategy="exact")
    result = RandomSampler(n_samples=200, seed=0).solve(problem_N8K4)
    table = problem_N8K4.compare([result], reference_point=ref)
    np.testing.assert_array_equal(table.reference_point, ref)


def test_problem_compare_to_string(problem_N8K4):
    from qportfolio.solvers.random import RandomSampler

    result = RandomSampler(n_samples=200, seed=0).solve(problem_N8K4)
    table = problem_N8K4.compare([result])
    s = table.to_string()
    assert "random" in s
