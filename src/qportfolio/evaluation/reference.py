"""Reference point computation strategies for hypervolume evaluation.

The reference point must be fixed *before* any solver runs and must be
strictly dominated by every feasible portfolio — otherwise the hypervolume
of solver results is not comparable across solvers or runs.
"""

from __future__ import annotations

from enum import Enum
from math import comb
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qportfolio.problems.portfolio import PortfolioProblem

from qportfolio.evaluation.hypervolume import NormalizedHypervolume


class ReferenceStrategy(str, Enum):
    EXACT = "exact"
    """Enumerate all feasible portfolios (small N) or use MILP to find bounds."""
    CONSERVATIVE = "conservative"
    """Valid lower bounds computed without full enumeration; may be loose."""
    MANUAL = "manual"
    """User-supplied reference point."""


def compute_reference_point(
    problem: PortfolioProblem,
    strategy: ReferenceStrategy | str = ReferenceStrategy.EXACT,
    margin_factor: float = 0.05,
    manual_point: np.ndarray | None = None,
    milp_time_limit: float = 30.0,
) -> np.ndarray:
    """Compute a reference point strictly dominated by all feasible portfolios.

    Parameters
    ----------
    problem : PortfolioProblem
    strategy : how to find the lower bound of each objective.
        - ``"exact"``: enumerate (small N) or MILP (large N).
        - ``"conservative"``: heuristic lower bounds, always valid but possibly loose.
        - ``"manual"``: use ``manual_point`` directly (no validation).
    margin_factor : amount to subtract below the minimum feasible objective.
        Reference point at ``L_j - margin_factor * |L_j| - 1e-6`` per dimension.
    manual_point : required when strategy is ``"manual"``.
    milp_time_limit : per-objective time budget for MILP-based bound computation.

    Returns
    -------
    np.ndarray of shape (3,) — reference point in objective space.
    """
    strategy = ReferenceStrategy(strategy)

    if strategy is ReferenceStrategy.MANUAL:
        if manual_point is None:
            raise ValueError("manual_point must be provided when strategy='manual'.")
        return np.asarray(manual_point, dtype=float)

    if strategy is ReferenceStrategy.EXACT:
        lower = _exact_lower_bounds(problem, milp_time_limit)
    else:  # CONSERVATIVE
        lower = _conservative_lower_bounds(problem)

    # Strictly dominated: L_j - margin*|L_j| - epsilon
    ref = lower - margin_factor * np.abs(lower) - 1e-6
    return ref


def _exact_lower_bounds(
    problem: PortfolioProblem,
    milp_time_limit: float,
) -> np.ndarray:
    """Find per-objective minimum over all feasible portfolios.

    Uses enumeration when C(N,K) ≤ 1,000,000, MILP otherwise.
    """
    from math import comb

    N = len(problem.tickers)
    K = problem.K

    if comb(N, K) <= 1_000_000:
        return _enumerate_lower_bounds(problem)
    else:
        return _milp_lower_bounds(problem, milp_time_limit)


def _enumerate_lower_bounds(problem: PortfolioProblem) -> np.ndarray:
    """Enumerate all feasible portfolios and return per-objective minimums."""
    all_x = problem.all_feasible_portfolios()
    all_objs = problem.evaluate_batch(all_x.astype(float))
    return all_objs.min(axis=0)


def _milp_lower_bounds(
    problem: PortfolioProblem,
    milp_time_limit: float,
) -> np.ndarray:
    """Use MILP to compute objective lower bounds when enumeration is infeasible."""
    from qportfolio.optimization.milp import PortfolioMILP

    milp = PortfolioMILP(problem.mu, problem.sigma, problem.D, problem.K)
    lower_bounds, _ = milp.compute_objective_bounds(time_limit_per_bound=milp_time_limit)
    return lower_bounds


def _conservative_lower_bounds(problem: PortfolioProblem) -> np.ndarray:
    """Heuristic lower bounds — always valid, may be loose.

    f1 (negative risk): at most 0 (variance ≥ 0, so -variance ≤ 0)
    f2 (return): minimum possible = K * min(mu)
    f3 (diversification): minimum = 0 (K stocks all in same sector)
    """
    len(problem.tickers)
    # f1 lower bound: worst case = -(max diagonal of sigma) * K (very loose)
    max_var = float(np.diag(problem.sigma).max())
    f1_lower = -(max_var * problem.K**2)

    # f2 lower bound
    f2_lower = float(np.sort(problem.mu)[: problem.K].sum())

    # f3 lower bound: always 0 (could pick K stocks all in one sector)
    f3_lower = 0.0

    return np.array([f1_lower, f2_lower, f3_lower])


def build_normalizer(
    problem: PortfolioProblem,
    strategy: ReferenceStrategy | str = ReferenceStrategy.EXACT,
    milp_time_limit: float = 30.0,
) -> NormalizedHypervolume:  # type: ignore[name-defined]  # forward ref
    """Construct a NormalizedHypervolume for the given problem.

    The normalizer shift/scale is derived from the full objective range
    [lower_bound, upper_bound] over all feasible portfolios.

    Parameters
    ----------
    problem : PortfolioProblem
    strategy : used to find bounds (same options as ``compute_reference_point``).
    milp_time_limit : per-objective MILP budget for large problems.
    """

    N = len(problem.tickers)
    K = problem.K
    strategy = ReferenceStrategy(strategy)

    if strategy is ReferenceStrategy.EXACT and comb(N, K) <= 1_000_000:
        all_x = problem.all_feasible_portfolios()
        all_objs = problem.evaluate_batch(all_x.astype(float))
        lower = all_objs.min(axis=0)
        upper = all_objs.max(axis=0)
    elif strategy is ReferenceStrategy.EXACT:
        from qportfolio.optimization.milp import PortfolioMILP

        milp = PortfolioMILP(problem.mu, problem.sigma, problem.D, problem.K)
        lower, upper = milp.compute_objective_bounds(time_limit_per_bound=milp_time_limit)
    else:
        lower = _conservative_lower_bounds(problem)
        # Conservative upper: not tight, but always valid
        upper = lower + 1.0  # scale will be corrected later

    return NormalizedHypervolume.from_reference_range(lower, upper)
