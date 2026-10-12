"""Base solver interface and shared result types for qportfolio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qportfolio.optimization.pareto_archive import SolutionRecord


@dataclass
class OptimizationResult:
    """Standardized output from any solver.

    All solvers return this type so that the evaluation engine can compare
    them under a consistent framework.
    """

    algorithm: str
    problem_id: str
    status: str  # "optimal" | "feasible" | "timeout" | "failed"

    # Solutions
    solutions: list[SolutionRecord]  # all unique feasible solutions found
    nondominated_solutions: list[SolutionRecord]

    # Summary metrics
    hypervolume: float | None
    n_nondominated: int
    n_feasible: int  # unique feasible portfolios
    n_unique: int  # same as n_feasible
    n_evaluations: int  # total objective evaluations performed

    # Timing
    runtime: float  # total wall-clock seconds

    # History (HV vs time)
    history: list[dict[str, Any]]

    # Algorithm-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    classical: dict[str, Any] | None = None
    quantum: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dict for display."""
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "n_nondominated": self.n_nondominated,
            "n_feasible": self.n_feasible,
            "hypervolume": round(self.hypervolume, 6) if self.hypervolume is not None else None,
            "runtime_s": round(self.runtime, 3),
            "n_evaluations": self.n_evaluations,
        }


class BaseSolver(ABC):
    """Abstract base class for all qportfolio solvers.

    Subclasses must implement `solve(problem) -> OptimizationResult`.
    """

    @abstractmethod
    def solve(self, problem: Any) -> OptimizationResult:
        """Run the optimization algorithm on the given problem.

        Parameters
        ----------
        problem:
            A PortfolioProblem instance.

        Returns
        -------
        OptimizationResult
        """

    def _make_result(
        self,
        algorithm: str,
        problem: Any,
        archive: Any,
        n_evaluations: int,
        runtime: float,
        status: str,
        metadata: dict | None = None,
        classical: dict | None = None,
        quantum: dict | None = None,
    ) -> OptimizationResult:
        """Helper: build an OptimizationResult from a ParetoArchive."""
        nd_solutions = archive.nondominated()

        # Compute final hypervolume if reference point is set
        hv = None
        if archive._reference_point is not None and len(nd_solutions) > 0:
            try:
                hv = archive.hypervolume()
            except Exception:  # noqa: BLE001
                hv = None

        history = [
            {
                "elapsed_time": e.elapsed_time,
                "n_nondominated": e.n_nondominated,
                "n_feasible": e.n_feasible,
                "hypervolume": e.hypervolume,
            }
            for e in archive.history()
        ]

        return OptimizationResult(
            algorithm=algorithm,
            problem_id=problem.problem_id,
            status=status,
            solutions=archive.all_solutions(),
            nondominated_solutions=nd_solutions,
            hypervolume=hv,
            n_nondominated=archive.n_nondominated,
            n_feasible=archive.n_feasible,
            n_unique=archive.n_unique,
            n_evaluations=n_evaluations,
            runtime=runtime,
            history=history,
            metadata=metadata or {},
            classical=classical,
            quantum=quantum,
        )
