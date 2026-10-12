"""Timing utilities and solution discovery events for benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SolutionDiscoveryEvent:
    """Records the moment a new non-dominated solution was discovered."""

    elapsed_time: float
    """Wall-clock seconds since the solver started."""

    candidate_id: str | None
    """Optional identifier for this candidate (e.g., bitstring hash)."""

    objective_values: tuple[float, ...]
    """Objective vector at discovery time."""

    feasible: bool
    """Whether this candidate was feasible."""

    n_nondominated: int
    """Size of the non-dominated set after this update."""

    hypervolume: float | None
    """HV at this event (None if reference point not set or HV not computed yet)."""

    source: str | None = None
    """Solver/operator that produced this candidate."""


class SolverTimer:
    """Simple perf_counter-based timer, shared across solver and archive."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        self._start = time.perf_counter()
        self._running = True

    def elapsed(self) -> float:
        if not self._running:
            return 0.0
        return time.perf_counter() - self._start

    def stop(self) -> float:
        t = self.elapsed()
        self._running = False
        return t


@dataclass
class QAOATiming:
    """Breakdown of QAOA runtime into its components."""

    training_time: float = 0.0
    """Wall-clock seconds for classical/Julia parameter optimization."""

    sampling_time: float = 0.0
    """Wall-clock seconds for quantum circuit sampling across all weight vectors."""

    other_time: float = 0.0
    """Wall-clock seconds for overhead (archive updates, HV computation, etc.)."""

    n_repetitions: int = 1
    """Number of independent QAOA runs (for amortized cost)."""

    @property
    def t_cold(self) -> float:
        """Total time including full training cost."""
        return self.training_time + self.sampling_time + self.other_time

    @property
    def t_amortized(self) -> float:
        """Total time amortizing training over n_repetitions."""
        return self.training_time / self.n_repetitions + self.sampling_time + self.other_time

    def to_dict(self) -> dict:
        return {
            "training_time": self.training_time,
            "sampling_time": self.sampling_time,
            "other_time": self.other_time,
            "n_repetitions": self.n_repetitions,
            "t_cold": self.t_cold,
            "t_amortized": self.t_amortized,
        }
