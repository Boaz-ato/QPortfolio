"""Unit tests for NSGA-II solver."""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.optimization.pareto_archive import _dominates
from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.solvers.nsga2 import NSGA2Solver, _union_crossover

# ---------------------------------------------------------------------------
# Cardinality-preserving operator tests (no pymoo required)
# ---------------------------------------------------------------------------


def test_union_crossover_preserves_K():
    """Output of union crossover must always have exactly K selected stocks."""
    N, K = 10, 3
    rng = np.random.default_rng(0)
    for _ in range(50):
        p1 = np.zeros(N)
        p1[rng.choice(N, K, replace=False)] = 1.0
        p2 = np.zeros(N)
        p2[rng.choice(N, K, replace=False)] = 1.0
        child = _union_crossover(p1, p2, K, N, rng)
        assert child.sum() == K, f"child.sum()={child.sum()} != K={K}"
        assert set(np.unique(child)).issubset({0.0, 1.0}), "child not binary"


def test_union_crossover_child_from_parents():
    """When union has ≥ K members, child should only contain parent-selected stocks."""
    N, K = 8, 2
    rng = np.random.default_rng(5)
    p1 = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    p2 = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=float)
    union_set = {0, 1, 2, 3}
    for _ in range(20):
        child = _union_crossover(p1, p2, K, N, rng)
        selected = set(int(i) for i in np.where(child > 0.5)[0])
        assert selected.issubset(union_set), (
            f"Child selected {selected} outside parent union {union_set}"
        )


# ---------------------------------------------------------------------------
# NSGA-II solver tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def problem_N8K4():
    rng = np.random.default_rng(7)
    N, K = 8, 4
    tickers = [f"S{i:02d}" for i in range(N)]
    sectors = ["Tech", "Health", "Finance", "Energy", "Tech", "Health", "Finance", "Energy"]
    sector_map = {t: s for t, s in zip(tickers, sectors)}
    mu = rng.uniform(0.08, 0.28, N)
    A = rng.normal(0, 0.06, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)
    return PortfolioProblem.from_arrays(
        tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K, seed=7
    )


@pytest.fixture(scope="module")
def exact_pareto_n8k4(problem_N8K4):
    all_x = problem_N8K4.all_feasible_portfolios()
    all_objs = problem_N8K4.evaluate_batch(all_x.astype(float))
    nd_mask = np.ones(len(all_x), dtype=bool)
    for i in range(len(all_x)):
        for j in range(len(all_x)):
            if i != j and _dominates(all_objs[j], all_objs[i]):
                nd_mask[i] = False
                break
    return all_x[nd_mask], all_objs[nd_mask]


def test_nsga2_result_structure(problem_N8K4):
    solver = NSGA2Solver(pop_size=20, n_generations=10, seed=0)
    result = solver.solve(problem_N8K4)
    assert result.algorithm == "nsga2"
    assert result.status in ("feasible", "failed")
    assert result.n_nondominated >= 0
    assert result.runtime >= 0


def test_nsga2_all_solutions_feasible(problem_N8K4):
    solver = NSGA2Solver(pop_size=20, n_generations=5, seed=0)
    result = solver.solve(problem_N8K4)
    for sol in result.solutions:
        assert problem_N8K4.validator.is_feasible(sol.bitstring), (
            f"Infeasible: sum={sol.bitstring.sum()}"
        )


def test_nsga2_nd_subset_of_exact(problem_N8K4, exact_pareto_n8k4):
    _, exact_objs = exact_pareto_n8k4
    solver = NSGA2Solver(pop_size=30, n_generations=20, seed=42)
    result = solver.solve(problem_N8K4)
    for sol in result.nondominated_solutions:
        dominated = any(_dominates(exact_objs[k], sol.objectives) for k in range(len(exact_objs)))
        assert not dominated


def test_nsga2_finds_nondominated(problem_N8K4):
    solver = NSGA2Solver(pop_size=30, n_generations=30, seed=0)
    result = solver.solve(problem_N8K4)
    assert result.n_nondominated >= 1


def test_nsga2_nd_solutions_internally_nondominated(problem_N8K4):
    """ND solutions reported by NSGA-II must not dominate each other."""
    solver = NSGA2Solver(pop_size=20, n_generations=10, seed=0)
    result = solver.solve(problem_N8K4)
    nd = result.nondominated_solutions
    for i, a in enumerate(nd):
        for j, b in enumerate(nd):
            if i != j:
                assert not _dominates(b.objectives, a.objectives), (
                    f"Solution {j} dominates {i}: {b.objectives} > {a.objectives}"
                )


def test_nsga2_classical_metadata(problem_N8K4):
    solver = NSGA2Solver(pop_size=20, n_generations=5, seed=0)
    result = solver.solve(problem_N8K4)
    assert result.classical is not None
    assert result.classical["pop_size"] == 20
    assert result.classical["n_generations"] == 5
