"""Randomized ε-Constraint Method (ε-CM) solver.

Implements the randomized ε-constraint method from the reference paper:
  - Sample weight vector c from the simplex
  - Sample ε thresholds for each objective from [L_j, U_j]
  - Solve the scalarized problem with ε constraints using HiGHS MILP

 This is the randomized variant that mirrors the QAOA weight-sampling approach.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from qportfolio.optimization.milp import PortfolioMILP
from qportfolio.optimization.pareto_archive import ParetoArchive
from qportfolio.optimization.weight_sampling import SimplexWeightSampler
from qportfolio.solvers.base import BaseSolver, OptimizationResult
from qportfolio.utils.logging import get_logger

_log = get_logger("solvers.epsilon")


class EpsilonCMSolver(BaseSolver):
    """Randomized ε-Constraint Method solver.

    Parameters
    ----------
    n_samples:
        Number of (weight, ε) draws.
    seed:
        Random seed for weight and ε sampling.
    time_limit_per_solve:
        Per-MILP time limit in seconds.
    total_time_limit:
        Overall time limit in seconds. None = no limit.
    epsilon_margin:
        Fraction of the objective range to exclude from the ε sampling interval.
        E.g. 0.05 → ε is sampled from [L + 0.05*(U-L), U - 0.05*(U-L)].
        Avoids trivially infeasible and trivially unconstrained problems.
    bound_time_limit:
        Time limit per objective-bound solve (seconds).
    hv_reference_point:
        Optional (3,) reference point for inline HV tracking.
    """

    def __init__(
        self,
        n_samples: int = 100,
        seed: int = 0,
        time_limit_per_solve: float = 30.0,
        total_time_limit: float | None = None,
        epsilon_margin: float = 0.05,
        bound_time_limit: float = 10.0,
        hv_reference_point: np.ndarray | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.seed = seed
        self.time_limit_per_solve = time_limit_per_solve
        self.total_time_limit = total_time_limit
        self.epsilon_margin = epsilon_margin
        self.bound_time_limit = bound_time_limit
        self.hv_reference_point = hv_reference_point

    def solve(self, problem: Any) -> OptimizationResult:
        milp = PortfolioMILP(problem.mu, problem.sigma, problem.D, problem.K)

        # Compute objective bounds once (used for ε sampling interval)
        _log.info("ε-CM: computing objective bounds...")
        lower_bounds, upper_bounds = milp.compute_objective_bounds(
            time_limit_per_bound=self.bound_time_limit
        )

        sampler = SimplexWeightSampler(n_objectives=3)
        rng = np.random.default_rng(self.seed)

        archive = ParetoArchive(problem, hv_cadence=5)
        if self.hv_reference_point is not None:
            archive.set_reference_point(self.hv_reference_point)

        t_start = time.perf_counter()
        n_evaluations = 0
        n_feasible_solves = 0
        n_infeasible = 0
        weight_vectors_used: list[np.ndarray] = []
        epsilon_vectors_used: list[np.ndarray] = []

        bar = tqdm(range(self.n_samples), desc="ε-CM samples", unit="sample", leave=False)
        for idx in bar:
            if self.total_time_limit is not None:
                elapsed = time.perf_counter() - t_start
                if elapsed >= self.total_time_limit:
                    _log.info("ε-CM: total time limit reached at iteration %d", idx)
                    break

            # Sample weight vector
            w = sampler.sample_one(rng)
            # Sample ε thresholds from [L + margin*(U-L), U - margin*(U-L)]
            eps = self._sample_epsilons(lower_bounds, upper_bounds, rng)

            weight_vectors_used.append(w)
            epsilon_vectors_used.append(eps)

            result = milp.solve_epsilon_cm(
                w,
                eps,
                time_limit=self.time_limit_per_solve,
            )

            if result.x is not None and result.objectives is not None:
                n_evaluations += 1
                n_feasible_solves += 1
                elapsed = time.perf_counter() - t_start
                objs = np.array(result.objectives)
                archive.add(result.x, objs, elapsed_time=elapsed, source=f"eps_w{idx}")
            else:
                n_infeasible += 1

            bar.set_postfix(nd=archive.n_nondominated, infeasible=n_infeasible)

        runtime = time.perf_counter() - t_start
        status = "feasible" if archive.n_nondominated > 0 else "failed"

        _log.info(
            "ε-CM: %d samples, %d feasible, %d infeasible, %d non-dominated, t=%.2fs",
            self.n_samples,
            n_feasible_solves,
            n_infeasible,
            archive.n_nondominated,
            runtime,
        )

        return self._make_result(
            algorithm="epsilon_cm",
            problem=problem,
            archive=archive,
            n_evaluations=n_evaluations,
            runtime=runtime,
            status=status,
            classical={
                "n_samples": self.n_samples,
                "n_feasible_solves": n_feasible_solves,
                "n_infeasible": n_infeasible,
                "lower_bounds": lower_bounds.tolist(),
                "upper_bounds": upper_bounds.tolist(),
                "epsilon_margin": self.epsilon_margin,
                "seed": self.seed,
            },
        )

    def _sample_epsilons(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample ε thresholds uniformly from the interior of [L, U].

        A margin is applied to avoid trivially infeasible and trivially
        unconstrained problems. If L ≈ U (degenerate objective), ε = L.
        """
        eps = np.empty(3)
        m = self.epsilon_margin
        for j in range(3):
            lo = lower[j]
            hi = upper[j]
            span = hi - lo
            if span < 1e-10:
                eps[j] = lo
            else:
                eps[j] = rng.uniform(lo + m * span, hi - m * span)
        return eps
