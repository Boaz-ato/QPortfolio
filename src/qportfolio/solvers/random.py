"""Random sampling baseline solver.

Samples portfolios uniformly at random from the feasible set
{x ∈ {0,1}^N : sum(x_i) = K}, evaluates their objectives, and
maintains the Pareto archive.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from qportfolio.optimization.pareto_archive import ParetoArchive
from qportfolio.solvers.base import BaseSolver, OptimizationResult
from qportfolio.utils.logging import get_logger

_log = get_logger("solvers.random")


class RandomSampler(BaseSolver):
    """Uniform random portfolio sampler.

    Parameters
    ----------
    n_samples:
        Number of portfolios to draw.
    seed:
        Random seed.
    unique:
        If True, avoid submitting duplicate portfolios to the archive.
        (Duplicates are tracked but not re-evaluated.)
    batch_size:
        Number of portfolios to evaluate in one vectorized batch.
    time_limit:
        Wall-clock time limit in seconds. None = no limit.
    hv_reference_point:
        Optional (3,) reference point for inline HV tracking.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seed: int = 0,
        unique: bool = True,
        batch_size: int = 512,
        time_limit: float | None = None,
        hv_reference_point: np.ndarray | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.seed = seed
        self.unique = unique
        self.batch_size = batch_size
        self.time_limit = time_limit
        self.hv_reference_point = hv_reference_point

    def solve(self, problem: Any) -> OptimizationResult:
        """Run random sampling on the given PortfolioProblem.

        Parameters
        ----------
        problem:
            A PortfolioProblem instance.
        """
        N, K = problem.N, problem.K
        rng = np.random.default_rng(self.seed)

        archive = ParetoArchive(problem, hv_cadence=20)
        if self.hv_reference_point is not None:
            archive.set_reference_point(self.hv_reference_point)

        t_start = time.perf_counter()
        n_evaluations = 0
        n_draws = 0
        seen: set[bytes] = set()  # for unique mode

        remaining = self.n_samples
        bar = tqdm(
            total=self.n_samples, desc="Random samples", unit="sample", unit_scale=True, leave=False
        )
        while remaining > 0:
            if self.time_limit is not None and (time.perf_counter() - t_start) >= self.time_limit:
                break

            batch = min(self.batch_size, remaining)
            # Draw `batch` random K-portfolios
            X = np.zeros((batch, N), dtype=np.float64)
            for b in range(batch):
                idx = rng.choice(N, size=K, replace=False)
                X[b, idx] = 1.0

            n_draws += batch

            if self.unique:
                # Filter duplicates not yet seen globally
                mask = []
                for b in range(batch):
                    key = X[b].astype(np.uint8).tobytes()
                    if key not in seen:
                        seen.add(key)
                        mask.append(b)
                if not mask:
                    remaining -= batch
                    bar.update(batch)
                    continue
                X_eval = X[mask]
            else:
                X_eval = X

            # Batch objective evaluation
            objs = problem.evaluate_batch(X_eval)
            n_evaluations += len(X_eval)

            elapsed = time.perf_counter() - t_start
            archive.add_batch(X_eval, objs, elapsed_time=elapsed, source="random")

            remaining -= batch
            bar.update(batch)
            bar.set_postfix(nd=archive.n_nondominated, unique=n_evaluations)
        bar.close()

        runtime = time.perf_counter() - t_start
        status = "feasible" if archive.n_nondominated > 0 else "failed"

        _log.info(
            "RandomSampler: %d draws, %d unique evaluated, %d non-dominated, t=%.2fs",
            n_draws,
            n_evaluations,
            archive.n_nondominated,
            runtime,
        )

        return self._make_result(
            algorithm="random",
            problem=problem,
            archive=archive,
            n_evaluations=n_evaluations,
            runtime=runtime,
            status=status,
            classical={
                "n_draws": n_draws,
                "n_unique_evaluated": n_evaluations,
                "unique_mode": self.unique,
                "seed": self.seed,
            },
        )
