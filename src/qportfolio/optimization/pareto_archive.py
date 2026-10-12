"""Shared Pareto archive used by all solvers.

Maintains a set of feasible, unique, non-dominated solutions discovered
during optimization. All solvers feed into the same archive so that
hypervolume comparisons are mathematically fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.indicators.hv import HV


@dataclass
class SolutionRecord:
    """A single feasible portfolio solution."""

    bitstring: np.ndarray  # shape (N,) uint8/float binary
    selected_tickers: list[str]
    objectives: np.ndarray  # shape (3,) [f1, f2, f3] maximization
    pareto_rank: int | None  # 0 = non-dominated in current archive
    first_seen_time: float  # elapsed seconds since solver start
    source: str  # e.g. "random", "wsm_iter_42"
    frequency: int = 1  # times this exact bitstring was sampled

    @property
    def risk(self) -> float:
        """Portfolio risk = -f1 (positive value)."""
        return float(-self.objectives[0])

    @property
    def expected_return(self) -> float:
        return float(self.objectives[1])

    @property
    def diversification(self) -> float:
        return float(self.objectives[2])


@dataclass
class DiscoveryEvent:
    """Records the state of the archive when a new non-dominated solution was found."""

    elapsed_time: float
    n_nondominated: int
    n_feasible: int
    hypervolume: float | None


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b (all objectives >= and at least one >)."""
    return bool(np.all(a >= b) and np.any(a > b))


class ParetoArchive:
    """Tracks feasible, unique, non-dominated solutions.

    Parameters
    ----------
    problem:
        The PortfolioProblem instance (used for feasibility checks and ticker names).
    hv_cadence:
        Compute hypervolume every hv_cadence new non-dominated solutions.
        Set to 0 to disable inline hypervolume tracking (compute on demand only).
    """

    def __init__(self, problem: Any, hv_cadence: int = 10) -> None:
        self._problem = problem
        self._hv_cadence = hv_cadence

        # All feasible solutions (deduped by bitstring hash)
        self._all: dict[bytes, SolutionRecord] = {}
        # Non-dominated subset (rebuilt on each update)
        self._nondominated: list[SolutionRecord] = []
        # Discovery history
        self._history: list[DiscoveryEvent] = []
        # Internal counter for HV cadence
        self._nd_since_last_hv = 0
        # Reference point (set externally or lazily)
        self._reference_point: np.ndarray | None = None
        # Cached (M, 3) array of ND objectives for vectorized domination checks
        self._nd_objs: np.ndarray = np.empty((0, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Adding solutions
    # ------------------------------------------------------------------

    def add(
        self,
        x: np.ndarray,
        objectives: np.ndarray,
        elapsed_time: float = 0.0,
        source: str = "",
    ) -> bool:
        """Add a single solution. Returns True if it joined the non-dominated set."""
        x = np.asarray(x, dtype=np.float64)
        objectives = np.asarray(objectives, dtype=np.float64)

        # Feasibility check
        if not self._problem.validator.is_feasible(x):
            return False

        key = x.astype(np.uint8).tobytes()

        # Duplicate → increment frequency only
        if key in self._all:
            self._all[key].frequency += 1
            return False

        # New feasible solution
        tickers = [self._problem.tickers[i] for i in range(self._problem.N) if x[i] > 0.5]
        record = SolutionRecord(
            bitstring=x.copy(),
            selected_tickers=tickers,
            objectives=objectives.copy(),
            pareto_rank=None,
            first_seen_time=elapsed_time,
            source=source,
        )
        self._all[key] = record

        # Update non-dominated set
        is_nd = self._update_nondominated(record)
        return is_nd

    def add_batch(
        self,
        X: np.ndarray,
        objectives_batch: np.ndarray,
        elapsed_time: float = 0.0,
        source: str = "",
    ) -> int:
        """Add a batch of solutions. Returns count of new non-dominated solutions."""
        n_new_nd = 0
        for i in range(len(X)):
            if self.add(X[i], objectives_batch[i], elapsed_time, source):
                n_new_nd += 1
        return n_new_nd

    # ------------------------------------------------------------------
    # Internal non-dominated set maintenance
    # ------------------------------------------------------------------

    def _update_nondominated(self, new_record: SolutionRecord) -> bool:
        """Check if new_record is non-dominated; update the ND set accordingly."""
        new_obj = new_record.objectives

        if len(self._nondominated) > 0:
            # Vectorized check: is new_obj dominated by any existing ND solution?
            dominated_by = np.all(self._nd_objs >= new_obj, axis=1) & np.any(
                self._nd_objs > new_obj, axis=1
            )
            if dominated_by.any():
                return False  # dominated, skip

            # Vectorized: which existing ND solutions are dominated by new_obj?
            new_dominates = np.all(new_obj >= self._nd_objs, axis=1) & np.any(
                new_obj > self._nd_objs, axis=1
            )
            keep_mask = ~new_dominates
            self._nondominated = [s for s, k in zip(self._nondominated, keep_mask) if k]
            self._nd_objs = np.vstack([self._nd_objs[keep_mask], new_obj])
        else:
            self._nd_objs = new_obj.reshape(1, 3)

        self._nondominated.append(new_record)
        new_record.pareto_rank = 0

        # Record discovery event (with optional hypervolume)
        self._nd_since_last_hv += 1
        hv = None
        if self._hv_cadence > 0 and self._nd_since_last_hv >= self._hv_cadence:
            hv = self._compute_hypervolume_internal()
            self._nd_since_last_hv = 0

        self._history.append(
            DiscoveryEvent(
                elapsed_time=new_record.first_seen_time,
                n_nondominated=len(self._nondominated),
                n_feasible=len(self._all),
                hypervolume=hv,
            )
        )
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def nondominated(self) -> list[SolutionRecord]:
        """Return the current list of non-dominated solutions."""
        return list(self._nondominated)

    def all_solutions(self) -> list[SolutionRecord]:
        """Return all feasible, unique solutions."""
        return list(self._all.values())

    @property
    def n_nondominated(self) -> int:
        return len(self._nondominated)

    @property
    def n_feasible(self) -> int:
        """Number of unique feasible solutions added so far."""
        return len(self._all)

    @property
    def n_unique(self) -> int:
        return len(self._all)

    @property
    def n_total_samples(self) -> int:
        """Total number of feasible bitstrings seen (including duplicates)."""
        return sum(r.frequency for r in self._all.values())

    def history(self) -> list[DiscoveryEvent]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Hypervolume
    # ------------------------------------------------------------------

    def set_reference_point(self, reference_point: np.ndarray) -> None:
        """Set the reference point for hypervolume computation."""
        self._reference_point = np.asarray(reference_point, dtype=np.float64)

    def hypervolume(self, reference_point: np.ndarray | None = None) -> float:
        """Compute the hypervolume indicator of the current non-dominated set.

        Parameters
        ----------
        reference_point:
            Shape (3,) reference point. Falls back to self._reference_point if None.

        Returns
        -------
        float
            Hypervolume value, or 0.0 if the archive is empty.
        """
        ref = reference_point if reference_point is not None else self._reference_point
        if ref is None:
            raise ValueError(
                "No reference point available. Call set_reference_point() first "
                "or pass reference_point to hypervolume()."
            )
        return self._compute_hypervolume(np.asarray(ref, dtype=np.float64))

    def _compute_hypervolume_internal(self) -> float | None:
        if self._reference_point is None or len(self._nondominated) == 0:
            return None
        try:
            return self._compute_hypervolume(self._reference_point)
        except Exception:  # noqa: BLE001
            return None

    def _compute_hypervolume(self, reference_point: np.ndarray) -> float:
        """Internal HV computation using pymoo."""
        if len(self._nondominated) == 0:
            return 0.0

        objs = np.array([s.objectives for s in self._nondominated])  # (M, 3) maximization

        # pymoo HV uses minimization convention.
        # Negate objectives and reference point for maximization.
        neg_ref = -reference_point
        ind = HV(ref_point=neg_ref)
        return float(ind(-objs))
