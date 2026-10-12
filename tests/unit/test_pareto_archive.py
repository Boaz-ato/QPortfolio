"""Unit tests for ParetoArchive.

Includes enumeration-based ground-truth tests: for N=8, K=4 (C(8,4)=70 portfolios)
we can compute the exact Pareto front and verify the archive against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.optimization.pareto_archive import ParetoArchive, _dominates

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_problem(small_problem_N5K2):
    return small_problem_N5K2


def make_archive(problem):
    return ParetoArchive(problem, hv_cadence=0)


# ---------------------------------------------------------------------------
# Dominance helper
# ---------------------------------------------------------------------------


def test_dominates_true():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.5])
    assert _dominates(a, b)


def test_dominates_false_equal():
    a = np.array([1.0, 2.0, 3.0])
    assert not _dominates(a, a)


def test_dominates_false_partial():
    a = np.array([2.0, 1.0, 3.0])
    b = np.array([1.0, 2.0, 2.5])
    assert not _dominates(a, b)
    assert not _dominates(b, a)


# ---------------------------------------------------------------------------
# Basic archive operations
# ---------------------------------------------------------------------------


def test_empty_archive(tiny_problem):
    arch = make_archive(tiny_problem)
    assert arch.n_nondominated == 0
    assert arch.n_feasible == 0


def test_add_feasible_solution(tiny_problem):
    arch = make_archive(tiny_problem)
    x = np.array([1, 1, 0, 0, 0], dtype=float)
    objs = np.array(tiny_problem.evaluate(x))
    added = arch.add(x, objs, elapsed_time=0.1, source="test")
    assert added  # first solution is always non-dominated
    assert arch.n_nondominated == 1
    assert arch.n_feasible == 1


def test_add_infeasible_rejected(tiny_problem):
    arch = make_archive(tiny_problem)
    x = np.array([1, 0, 0, 0, 0], dtype=float)  # K=1, not K=2
    objs = np.zeros(3)
    added = arch.add(x, objs)
    assert not added
    assert arch.n_feasible == 0


def test_add_duplicate_increments_frequency(tiny_problem):
    arch = make_archive(tiny_problem)
    x = np.array([1, 1, 0, 0, 0], dtype=float)
    objs = np.array(tiny_problem.evaluate(x))
    arch.add(x, objs)
    arch.add(x, objs)
    arch.add(x, objs)
    assert arch.n_feasible == 1
    sol = arch.all_solutions()[0]
    assert sol.frequency == 3


def test_dominated_solution_not_in_nd_set(tiny_problem):
    """Add a solution, then add a dominating one; first should be removed from ND."""
    arch = make_archive(tiny_problem)
    # All feasible portfolios for N=5, K=2
    all_x = tiny_problem.all_feasible_portfolios()
    all_objs = tiny_problem.evaluate_batch(all_x.astype(float))
    for i in range(len(all_x)):
        arch.add(all_x[i].astype(float), all_objs[i])
    # After adding all, archive should be non-empty
    assert arch.n_nondominated > 0
    # All nd solutions must truly be non-dominated among themselves
    nd = arch.nondominated()
    for i, sol_a in enumerate(nd):
        for j, sol_b in enumerate(nd):
            if i != j:
                assert not _dominates(sol_b.objectives, sol_a.objectives), (
                    f"Solution {j} dominates {i}: {sol_b.objectives} > {sol_a.objectives}"
                )


def test_nondominated_subset_of_feasible(tiny_problem):
    arch = make_archive(tiny_problem)
    all_x = tiny_problem.all_feasible_portfolios()
    all_objs = tiny_problem.evaluate_batch(all_x.astype(float))
    for i in range(len(all_x)):
        arch.add(all_x[i].astype(float), all_objs[i])
    assert arch.n_nondominated <= arch.n_feasible


def test_hypervolume_nonnegative(tiny_problem):
    arch = make_archive(tiny_problem)
    all_x = tiny_problem.all_feasible_portfolios()
    all_objs = tiny_problem.evaluate_batch(all_x.astype(float))
    for i in range(len(all_x)):
        arch.add(all_x[i].astype(float), all_objs[i])

    # Use a reference point strictly dominated by all solutions
    min_objs = all_objs.min(axis=0)
    ref_pt = min_objs - 0.1 * np.abs(min_objs) - 0.01
    hv = arch.hypervolume(ref_pt)
    assert hv >= 0.0


def test_hypervolume_increases_monotonically(tiny_problem):
    """Adding better solutions to the archive should never decrease HV."""
    arch = make_archive(tiny_problem)
    all_x = tiny_problem.all_feasible_portfolios()
    all_objs = tiny_problem.evaluate_batch(all_x.astype(float))

    min_objs = all_objs.min(axis=0)
    ref_pt = min_objs - 0.1 * np.abs(min_objs) - 0.01
    arch.set_reference_point(ref_pt)

    prev_hv = 0.0
    for i in range(len(all_x)):
        arch.add(all_x[i].astype(float), all_objs[i])
        hv = arch.hypervolume()
        assert hv >= prev_hv - 1e-12, f"HV decreased at step {i}: {hv} < {prev_hv}"
        prev_hv = hv


def test_hypervolume_no_reference_point_raises(tiny_problem):
    arch = make_archive(tiny_problem)
    x = np.array([1, 1, 0, 0, 0], dtype=float)
    objs = np.array(tiny_problem.evaluate(x))
    arch.add(x, objs)
    with pytest.raises(ValueError, match="reference point"):
        arch.hypervolume()


# ---------------------------------------------------------------------------
# add_batch
# ---------------------------------------------------------------------------


def test_add_batch_counts(tiny_problem):
    arch = make_archive(tiny_problem)
    all_x = tiny_problem.all_feasible_portfolios()
    all_objs = tiny_problem.evaluate_batch(all_x.astype(float))
    n_new = arch.add_batch(all_x.astype(float), all_objs)
    assert n_new >= 1  # at least one non-dominated solution found
    assert arch.n_feasible == len(all_x)


# ---------------------------------------------------------------------------
# GROUND TRUTH: N=8, K=4 exact Pareto front
# ---------------------------------------------------------------------------


@pytest.fixture
def exact_problem_N8K4():
    """Synthetic N=8, K=4 problem with a known, enumerable Pareto front."""
    from qportfolio.problems.portfolio import PortfolioProblem

    rng = np.random.default_rng(42)
    N, K = 8, 4
    tickers = [f"S{i}" for i in range(N)]
    sectors = ["Tech", "Health", "Finance", "Energy"] * 2
    sector_map = {t: s for t, s in zip(tickers, sectors)}
    mu = rng.uniform(0.05, 0.30, N)
    A = rng.normal(0, 0.05, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)
    return PortfolioProblem.from_arrays(tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K)


def _exact_pareto_front(problem):
    """Enumerate all feasible portfolios and compute the exact ND set."""
    all_x = problem.all_feasible_portfolios()
    all_objs = problem.evaluate_batch(all_x.astype(float))
    nd_mask = np.ones(len(all_x), dtype=bool)
    for i in range(len(all_x)):
        for j in range(len(all_x)):
            if i != j and _dominates(all_objs[j], all_objs[i]):
                nd_mask[i] = False
                break
    return all_x[nd_mask], all_objs[nd_mask]


def test_archive_nd_subset_of_exact_nd(exact_problem_N8K4):
    """Archive's ND set must be a subset of the exact Pareto front."""
    from math import comb

    assert comb(8, 4) == 70  # sanity check

    exact_x, exact_objs = _exact_pareto_front(exact_problem_N8K4)

    arch = make_archive(exact_problem_N8K4)
    all_x = exact_problem_N8K4.all_feasible_portfolios()
    all_objs = exact_problem_N8K4.evaluate_batch(all_x.astype(float))
    arch.add_batch(all_x.astype(float), all_objs)

    # Every archive ND solution must not be dominated by any exact ND solution
    for arch_sol in arch.nondominated():
        dominated_by_exact = any(
            _dominates(exact_objs[k], arch_sol.objectives) for k in range(len(exact_x))
        )
        assert not dominated_by_exact, (
            f"Archive ND solution {arch_sol.objectives} is dominated by exact ND front"
        )


def test_archive_finds_all_exact_nd(exact_problem_N8K4):
    """After adding all feasible portfolios, archive ND == exact ND (same objective vectors)."""
    exact_x, exact_objs = _exact_pareto_front(exact_problem_N8K4)

    arch = make_archive(exact_problem_N8K4)
    all_x = exact_problem_N8K4.all_feasible_portfolios()
    all_objs = exact_problem_N8K4.evaluate_batch(all_x.astype(float))
    arch.add_batch(all_x.astype(float), all_objs)

    arch_objs = np.array([s.objectives for s in arch.nondominated()])

    # Every exact ND solution must appear in the archive ND set
    # (matching on rounded objective vectors to avoid floating-point issues)
    for k in range(len(exact_objs)):
        match = any(np.allclose(exact_objs[k], arch_objs[m], atol=1e-10) for m in range(len(arch_objs)))
        assert match, f"Exact ND solution {exact_objs[k]} not found in archive ND set"
