"""Benchmarking harness for multi-solver comparison.

Provides three stopping-criterion modes:
  - ``fixed_time``: wall-clock budget (seconds)
  - ``fixed_evaluations``: number of unique feasible portfolios evaluated
  - ``fixed_samples``: raw candidates generated (including infeasible)

Usage::

    from qportfolio.evaluation.benchmark import BenchmarkSuite

    suite = BenchmarkSuite(
        problem,
        solvers=[RandomSampler(...), WSMSolver(...), NSGA2Solver(...)],
        mode="fixed_time",
        budget=60.0,
    )
    results = suite.run()
    table = suite.comparison_table(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qportfolio.evaluation.hypervolume import _pymoo_hv
from qportfolio.evaluation.reference import ReferenceStrategy, build_normalizer, compute_reference_point
from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.solvers.base import BaseSolver


class BenchmarkMode:
    FIXED_TIME = "fixed_time"
    FIXED_EVALUATIONS = "fixed_evaluations"
    FIXED_SAMPLES = "fixed_samples"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmarking run."""

    mode: str = BenchmarkMode.FIXED_TIME
    """Stopping criterion mode."""

    budget: float = 60.0
    """Budget value: seconds for fixed_time, count for fixed_evaluations/fixed_samples."""

    reference_point: np.ndarray | None = None
    """Reference point for HV computation. If None, computed from problem."""

    reference_strategy: str = "exact"
    """Strategy for auto-computing reference point: 'exact', 'conservative', 'manual'."""

    milp_time_limit: float = 30.0
    """Per-objective MILP time limit when computing reference via MILP."""

    compute_normalized_hv: bool = True
    """If True, also compute normalized HV using problem-level affine transform."""

    repeat: int = 1
    """Number of independent repetitions per solver (seeds: seed, seed+1, ...)."""


@dataclass
class SolverBenchmarkResult:
    """Summary statistics for one solver on one problem."""

    algorithm: str
    problem_id: str
    mode: str
    budget: float

    n_nondominated: int = 0
    n_feasible: int = 0
    n_evaluations: int = 0
    runtime: float = 0.0

    hypervolume: float | None = None
    normalized_hypervolume: float | None = None

    status: str = "failed"

    # Objective statistics over ND solutions
    nd_objectives_mean: np.ndarray | None = None
    nd_objectives_std: np.ndarray | None = None

    # Per-repetition results (when repeat > 1)
    repetition_results: list = field(default_factory=list)

    raw_result: Any = None
    """The full OptimizationResult from the solver."""


@dataclass
class ComparisonTable:
    """Structured comparison of multiple solvers."""

    problem_id: str
    mode: str
    budget: float
    reference_point: np.ndarray | None

    rows: list[dict]
    """Each row: {algorithm, n_nondominated, hypervolume, normalized_hv, runtime, ...}"""

    def to_string(self, float_fmt: str = ".4f") -> str:
        """Pretty-print the comparison table."""
        if not self.rows:
            return "No results."

        columns = list(self.rows[0].keys())
        widths = {
            c: max(
                len(c),
                max(
                    len(f"{r[c]:{float_fmt}}" if isinstance(r[c], float) else str(r[c]))
                    for r in self.rows
                ),
            )
            for c in columns
        }

        sep = " | "
        header = sep.join(c.ljust(widths[c]) for c in columns)
        divider = "-+-".join("-" * widths[c] for c in columns)
        lines = [header, divider]
        for row in self.rows:
            formatted = []
            for c in columns:
                v = row[c]
                if isinstance(v, float):
                    cell = f"{v:{float_fmt}}"
                elif v is None:
                    cell = "N/A"
                else:
                    cell = str(v)
                formatted.append(cell.ljust(widths[c]))
            lines.append(sep.join(formatted))
        return "\n".join(lines)


class BenchmarkSuite:
    """Runs a list of solvers on a problem and produces a comparison table.

    Parameters
    ----------
    problem : PortfolioProblem
    solvers : list of configured BaseSolver instances.
    config : BenchmarkConfig or keyword arguments forwarded to BenchmarkConfig.
    """

    def __init__(
        self,
        problem: PortfolioProblem,
        solvers: list[BaseSolver],
        config: BenchmarkConfig | None = None,
        **config_kwargs: Any,
    ) -> None:
        self.problem = problem
        self.solvers = solvers
        self.config = config if config is not None else BenchmarkConfig(**config_kwargs)
        self._reference_point: np.ndarray | None = None
        self._normalizer = None

    def _ensure_reference_point(self) -> np.ndarray:
        if self._reference_point is not None:
            return self._reference_point
        if self.config.reference_point is not None:
            self._reference_point = np.asarray(self.config.reference_point, dtype=float)
        else:
            self._reference_point = compute_reference_point(
                self.problem,
                strategy=ReferenceStrategy(self.config.reference_strategy),
                milp_time_limit=self.config.milp_time_limit,
            )
        return self._reference_point

    def _ensure_normalizer(self):
        if self._normalizer is not None:
            return self._normalizer

        self._normalizer = build_normalizer(
            self.problem,
            strategy=ReferenceStrategy(self.config.reference_strategy),
            milp_time_limit=self.config.milp_time_limit,
        )
        return self._normalizer

    def run(self) -> list[SolverBenchmarkResult]:
        """Run all solvers and collect results."""
        ref_pt = self._ensure_reference_point()
        normalizer = self._ensure_normalizer() if self.config.compute_normalized_hv else None

        benchmark_results = []
        for solver in self.solvers:
            bench = self._run_solver(solver, ref_pt, normalizer)
            benchmark_results.append(bench)
        return benchmark_results

    def _run_solver(
        self,
        solver: BaseSolver,
        ref_pt: np.ndarray,
        normalizer,
    ) -> SolverBenchmarkResult:
        """Run a single solver (with repetitions if configured)."""
        algo = getattr(solver, "algorithm_name", type(solver).__name__.lower())

        bench = SolverBenchmarkResult(
            algorithm=algo,
            problem_id=self.problem.problem_id,
            mode=self.config.mode,
            budget=self.config.budget,
        )

        rep_results = []
        for rep in range(self.config.repeat):
            if rep > 0 and hasattr(solver, "seed"):
                try:
                    solver.seed = solver.seed + rep  # type: ignore[attr-defined]
                except AttributeError:
                    pass

            t0 = time.perf_counter()
            result = solver.solve(self.problem)
            time.perf_counter() - t0

            rep_results.append(result)

        # Use the last result as the primary (or average across reps for stats)
        primary = rep_results[-1]
        bench.raw_result = primary
        bench.status = primary.status
        bench.n_nondominated = primary.n_nondominated
        bench.n_feasible = primary.n_feasible
        bench.n_evaluations = primary.n_evaluations
        bench.runtime = primary.runtime
        bench.repetition_results = rep_results

        # Compute hypervolumes
        if primary.n_nondominated > 0:
            nd_objs = np.array([s.objectives for s in primary.nondominated_solutions])

            try:
                bench.hypervolume = _pymoo_hv(nd_objs, ref_pt)
            except Exception:
                bench.hypervolume = None

            if normalizer is not None:
                try:
                    bench.normalized_hypervolume = normalizer.compute(nd_objs, ref_pt)
                except Exception:
                    bench.normalized_hypervolume = None

            bench.nd_objectives_mean = nd_objs.mean(axis=0)
            bench.nd_objectives_std = (
                nd_objs.std(axis=0) if len(nd_objs) > 1 else np.zeros(nd_objs.shape[1])
            )

        return bench

    def comparison_table(
        self,
        results: list[SolverBenchmarkResult] | None = None,
    ) -> ComparisonTable:
        """Build a ComparisonTable from benchmark results.

        If results is None, runs the suite first.
        """
        if results is None:
            results = self.run()

        ref_pt = self._ensure_reference_point()
        rows = []
        for r in results:
            rows.append(
                {
                    "algorithm": r.algorithm,
                    "status": r.status,
                    "n_nondominated": r.n_nondominated,
                    "n_feasible": r.n_feasible,
                    "n_evaluations": r.n_evaluations,
                    "hypervolume": r.hypervolume,
                    "normalized_hv": r.normalized_hypervolume,
                    "runtime_s": r.runtime,
                }
            )

        return ComparisonTable(
            problem_id=self.problem.problem_id,
            mode=self.config.mode,
            budget=self.config.budget,
            reference_point=ref_pt,
            rows=rows,
        )
