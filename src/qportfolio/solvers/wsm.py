"""Weighted Sum Method (WSM) solver.

Scalarizes the three portfolio objectives using uniformly sampled weights
from the 3-simplex (Dirichlet(1,1,1)) and solves each weighted subproblem
as a MILP using HiGHS via scipy.

Reference: every scalarized optimum is a supported Pareto-optimal solution.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from qportfolio.optimization.milp import PortfolioMILP
from qportfolio.optimization.pareto_archive import ParetoArchive
from qportfolio.optimization.weight_sampling import SimplexWeightSampler
from qportfolio.solvers.base import BaseSolver, OptimizationResult
from qportfolio.utils.logging import get_logger

_log = get_logger("solvers.wsm")


class WSMSolver(BaseSolver):
    """Weighted Sum Method solver using the HiGHS MILP backend.

    Parameters
    ----------
    n_weights:
        Number of weight vectors (MILP solves) to run.
    seed:
        Random seed for weight sampling.
    optimality:
        'global' — each MILP is solved to certified optimality.
        'time_limit' — allow early termination; keep all feasible incumbents.
    time_limit_per_solve:
        Per-solve time limit in seconds (only used when optimality='time_limit').
    total_time_limit:
        Overall time limit in seconds. None = no limit.
    mip_rel_gap:
        Relative MIP gap tolerance for each solve.
    hv_reference_point:
        Optional (3,) reference point for inline HV tracking.
    """

    def __init__(
        self,
        n_weights: int = 100,
        seed: int = 0,
        optimality: Literal["global", "time_limit"] = "global",
        time_limit_per_solve: float | None = 30.0,
        total_time_limit: float | None = None,
        mip_rel_gap: float = 1e-4,
        hv_reference_point: np.ndarray | None = None,
    ) -> None:
        self.n_weights = n_weights
        self.seed = seed
        self.optimality = optimality
        self.time_limit_per_solve = time_limit_per_solve if optimality == "time_limit" else None
        self.total_time_limit = total_time_limit
        self.mip_rel_gap = mip_rel_gap
        self.hv_reference_point = hv_reference_point

    def solve(self, problem: Any) -> OptimizationResult:
        milp = PortfolioMILP(problem.mu, problem.sigma, problem.D, problem.K)
        sampler = SimplexWeightSampler(n_objectives=3)
        weight_vectors = sampler.sample(self.n_weights, seed=self.seed)

        archive = ParetoArchive(problem, hv_cadence=5)
        if self.hv_reference_point is not None:
            archive.set_reference_point(self.hv_reference_point)

        t_start = time.perf_counter()
        n_evaluations = 0
        n_optimal = 0
        n_feasible_solves = 0
        n_infeasible = 0

        bar = tqdm(weight_vectors, desc="WSM weights", unit="weight", leave=False)
        for idx, w in enumerate(bar):
            if self.total_time_limit is not None:
                elapsed = time.perf_counter() - t_start
                if elapsed >= self.total_time_limit:
                    _log.info("WSM: total time limit reached at iteration %d", idx)
                    break

            result = milp.solve_wsm(
                w,
                time_limit=self.time_limit_per_solve,
                mip_rel_gap=self.mip_rel_gap,
            )

            if result.x is not None and result.objectives is not None:
                n_evaluations += 1
                if result.status == "optimal":
                    n_optimal += 1
                else:
                    n_feasible_solves += 1

                elapsed = time.perf_counter() - t_start
                objs = np.array(result.objectives)
                archive.add(result.x, objs, elapsed_time=elapsed, source=f"wsm_w{idx}")
            else:
                n_infeasible += 1

            bar.set_postfix(nd=archive.n_nondominated, infeasible=n_infeasible)

        runtime = time.perf_counter() - t_start
        status = (
            "optimal"
            if n_optimal == self.n_weights
            else ("feasible" if archive.n_nondominated > 0 else "failed")
        )

        _log.info(
            "WSM: %d weights, %d optimal, %d feasible, %d infeasible, %d non-dominated, t=%.2fs",
            self.n_weights,
            n_optimal,
            n_feasible_solves,
            n_infeasible,
            archive.n_nondominated,
            runtime,
        )

        return self._make_result(
            algorithm="wsm",
            problem=problem,
            archive=archive,
            n_evaluations=n_evaluations,
            runtime=runtime,
            status=status,
            classical={
                "n_weights": self.n_weights,
                "n_optimal": n_optimal,
                "n_feasible_solves": n_feasible_solves,
                "n_infeasible": n_infeasible,
                "optimality": self.optimality,
                "mip_rel_gap": self.mip_rel_gap,
                "seed": self.seed,
            },
        )
