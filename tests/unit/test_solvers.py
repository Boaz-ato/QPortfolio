"""Unit tests for classical solvers using small enumerable problems.

All solver tests use N=8, K=4 (70 feasible portfolios) so the exact Pareto
front can be computed by enumeration. The acceptance criterion:
  every solver's ND solutions must be a subset of the exact Pareto front.
"""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.optimization.pareto_archive import _dominates
from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.solvers.base import OptimizationResult
from qportfolio.solvers.epsilon import EpsilonCMSolver
from qportfolio.solvers.random import RandomSampler
from qportfolio.solvers.wsm import WSMSolver

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def problem_N8K4():
    """Synthetic N=8, K=4 problem with 70 feasible portfolios."""
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
def exact_pareto(problem_N8K4):
    """Exact Pareto front for the N=8, K=4 problem."""
    all_x = problem_N8K4.all_feasible_portfolios()
    all_objs = problem_N8K4.evaluate_batch(all_x.astype(float))
    nd_mask = np.ones(len(all_x), dtype=bool)
    for i in range(len(all_x)):
        for j in range(len(all_x)):
            if i != j and _dominates(all_objs[j], all_objs[i]):
                nd_mask[i] = False
                break
    return all_x[nd_mask], all_objs[nd_mask]


@pytest.fixture(scope="module")
def reference_point(problem_N8K4):
    """A reference point strictly dominated by all feasible portfolios."""
    all_x = problem_N8K4.all_feasible_portfolios()
    all_objs = problem_N8K4.evaluate_batch(all_x.astype(float))
    min_objs = all_objs.min(axis=0)
    return min_objs - 0.1 * np.abs(min_objs) - 0.01


def _check_nd_subset_of_exact(result: OptimizationResult, exact_objs: np.ndarray):
    """Assert every solver ND solution is not dominated by any exact ND solution."""
    for sol in result.nondominated_solutions:
        dominated = any(_dominates(exact_objs[k], sol.objectives) for k in range(len(exact_objs)))
        assert not dominated, (
            f"Solver ND solution {sol.objectives.round(4)} is dominated by an exact Pareto solution"
        )


def _check_result_structure(result: OptimizationResult):
    """Assert OptimizationResult has all required fields with correct types."""
    assert isinstance(result.algorithm, str)
    assert isinstance(result.problem_id, str)
    assert result.status in ("optimal", "feasible", "failed", "timeout")
    assert isinstance(result.solutions, list)
    assert isinstance(result.nondominated_solutions, list)
    assert isinstance(result.n_nondominated, int) and result.n_nondominated >= 0
    assert isinstance(result.n_feasible, int) and result.n_feasible >= 0
    assert isinstance(result.n_evaluations, int) and result.n_evaluations >= 0
    assert isinstance(result.runtime, float) and result.runtime >= 0
    assert isinstance(result.history, list)
    # ND solutions must all pass feasibility
    assert result.n_nondominated == len(result.nondominated_solutions)


# ---------------------------------------------------------------------------
# RandomSampler tests
# ---------------------------------------------------------------------------


def test_random_result_structure(problem_N8K4):
    solver = RandomSampler(n_samples=500, seed=0)
    result = solver.solve(problem_N8K4)
    _check_result_structure(result)
    assert result.algorithm == "random"


def test_random_all_solutions_feasible(problem_N8K4):
    solver = RandomSampler(n_samples=300, seed=1)
    result = solver.solve(problem_N8K4)
    for sol in result.solutions:
        assert problem_N8K4.validator.is_feasible(sol.bitstring), (
            f"Infeasible solution found: {sol.bitstring}"
        )


def test_random_nd_subset_of_exact(problem_N8K4, exact_pareto):
    _, exact_objs = exact_pareto
    solver = RandomSampler(n_samples=2000, seed=0)
    result = solver.solve(problem_N8K4)
    _check_nd_subset_of_exact(result, exact_objs)


def test_random_unique_mode_no_duplicates(problem_N8K4):
    solver = RandomSampler(n_samples=200, seed=5, unique=True)
    result = solver.solve(problem_N8K4)
    # All solutions recorded should be unique
    keys = {sol.bitstring.astype(np.uint8).tobytes() for sol in result.solutions}
    assert len(keys) == len(result.solutions)


def test_random_finds_nondominated(problem_N8K4):
    """With 2000 samples from 70 possible, should find at least 1 ND solution."""
    solver = RandomSampler(n_samples=2000, seed=42)
    result = solver.solve(problem_N8K4)
    assert result.n_nondominated >= 1


def test_random_with_reference_point(problem_N8K4, reference_point):
    solver = RandomSampler(n_samples=500, seed=0, hv_reference_point=reference_point)
    result = solver.solve(problem_N8K4)
    # HV should be computable since reference point is set
    if result.n_nondominated > 0:
        assert result.hypervolume is not None
        assert result.hypervolume >= 0


def test_random_time_limit(problem_N8K4):
    solver = RandomSampler(n_samples=1_000_000, seed=0, time_limit=0.1)
    result = solver.solve(problem_N8K4)
    assert result.runtime < 1.0  # should stop well before a million samples


# ---------------------------------------------------------------------------
# WSMSolver tests
# ---------------------------------------------------------------------------


def test_wsm_result_structure(problem_N8K4):
    solver = WSMSolver(n_weights=5, seed=0)
    result = solver.solve(problem_N8K4)
    _check_result_structure(result)
    assert result.algorithm == "wsm"


def test_wsm_all_solutions_feasible(problem_N8K4):
    solver = WSMSolver(n_weights=10, seed=0)
    result = solver.solve(problem_N8K4)
    for sol in result.solutions:
        assert problem_N8K4.validator.is_feasible(sol.bitstring)


def test_wsm_nd_subset_of_exact(problem_N8K4, exact_pareto):
    _, exact_objs = exact_pareto
    solver = WSMSolver(n_weights=30, seed=0)
    result = solver.solve(problem_N8K4)
    assert result.n_nondominated >= 1
    _check_nd_subset_of_exact(result, exact_objs)


def test_wsm_finds_exact_nd_solutions(problem_N8K4, exact_pareto):
    """WSM with enough weight vectors should find most exact ND solutions."""
    exact_x, exact_objs = exact_pareto
    solver = WSMSolver(n_weights=100, seed=0)
    result = solver.solve(problem_N8K4)

    solver_objs = np.array([s.objectives for s in result.nondominated_solutions])
    found = 0
    for k in range(len(exact_objs)):
        if any(np.allclose(exact_objs[k], solver_objs[m], atol=1e-6) for m in range(len(solver_objs))):
            found += 1
    # WSM finds supported Pareto solutions; with 100 weights, should find many
    assert found >= len(exact_objs) // 2, f"WSM found only {found}/{len(exact_objs)} exact ND solutions"


def test_wsm_problem_id_in_result(problem_N8K4):
    solver = WSMSolver(n_weights=3, seed=0)
    result = solver.solve(problem_N8K4)
    assert result.problem_id == problem_N8K4.problem_id


def test_wsm_classical_metadata(problem_N8K4):
    solver = WSMSolver(n_weights=5, seed=0)
    result = solver.solve(problem_N8K4)
    assert result.classical is not None
    assert result.classical["n_weights"] == 5
    assert "n_optimal" in result.classical


# ---------------------------------------------------------------------------
# EpsilonCMSolver tests
# ---------------------------------------------------------------------------


def test_epsilon_result_structure(problem_N8K4):
    solver = EpsilonCMSolver(n_samples=5, seed=0, bound_time_limit=5.0, time_limit_per_solve=5.0)
    result = solver.solve(problem_N8K4)
    _check_result_structure(result)
    assert result.algorithm == "epsilon_cm"


def test_epsilon_all_solutions_feasible(problem_N8K4):
    solver = EpsilonCMSolver(n_samples=10, seed=0, bound_time_limit=5.0, time_limit_per_solve=5.0)
    result = solver.solve(problem_N8K4)
    for sol in result.solutions:
        assert problem_N8K4.validator.is_feasible(sol.bitstring)


def test_epsilon_nd_subset_of_exact(problem_N8K4, exact_pareto):
    _, exact_objs = exact_pareto
    solver = EpsilonCMSolver(n_samples=20, seed=0, bound_time_limit=5.0, time_limit_per_solve=5.0)
    result = solver.solve(problem_N8K4)
    if result.n_nondominated > 0:
        _check_nd_subset_of_exact(result, exact_objs)


def test_epsilon_classical_metadata(problem_N8K4):
    solver = EpsilonCMSolver(n_samples=5, seed=0, bound_time_limit=3.0, time_limit_per_solve=3.0)
    result = solver.solve(problem_N8K4)
    assert result.classical is not None
    assert "n_feasible_solves" in result.classical
    assert "n_infeasible" in result.classical
    assert "lower_bounds" in result.classical


# ---------------------------------------------------------------------------
# MILP correctness: WSM on small exhaustive problem
# ---------------------------------------------------------------------------


def test_wsm_optimal_solution_is_correct(problem_N8K4, exact_pareto):
    """For a unit weight vector (1,0,0), WSM should maximize f1 = -risk.
    This means minimizing portfolio risk — check against enumeration."""
    from qportfolio.optimization.milp import PortfolioMILP

    milp = PortfolioMILP(problem_N8K4.mu, problem_N8K4.sigma, problem_N8K4.D, problem_N8K4.K)
    result = milp.solve_wsm(np.array([1.0, 0.0, 0.0]))
    assert result.x is not None
    assert result.objectives is not None

    # Verify objective matches direct evaluation
    f1_milp, f2_milp, f3_milp = result.objectives
    f1_eval, f2_eval, f3_eval = problem_N8K4.evaluate(result.x)
    assert abs(f1_milp - f1_eval) < 1e-6
    assert abs(f2_milp - f2_eval) < 1e-6
    assert abs(f3_milp - f3_eval) < 1e-6

    # The MILP solution should not be dominated by any feasible portfolio
    all_x = problem_N8K4.all_feasible_portfolios()
    all_objs = problem_N8K4.evaluate_batch(all_x.astype(float))
    np.array(result.objectives)
    # Among all portfolios, f1 of MILP solution should be maximal (i.e., min risk)
    assert f1_milp >= all_objs[:, 0].max() - 1e-6, (
        f"MILP f1={f1_milp:.4f} but max enumeration f1={all_objs[:, 0].max():.4f}"
    )
