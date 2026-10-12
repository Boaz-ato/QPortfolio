"""NSGA-II solver with cardinality-preserving operators.

Uses pymoo's NSGA-II implementation with custom binary sampling, mutation,
and crossover that all guarantee exactly K selected stocks per portfolio.

Key design choices:
  - Swap mutation: remove one selected stock, add one unselected stock.
  - Union crossover: take parents' combined selected set and randomly
    draw K stocks from it (filling from unselected stocks if needed).
  - External archive: tracks ALL non-dominated solutions across generations,
    not just the final population.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from tqdm import tqdm

from qportfolio.optimization.pareto_archive import ParetoArchive
from qportfolio.solvers.base import BaseSolver, OptimizationResult
from qportfolio.utils.logging import get_logger

_log = get_logger("solvers.nsga2")


class NSGA2Solver(BaseSolver):
    """NSGA-II multi-objective evolutionary solver.

    Parameters
    ----------
    pop_size:
        Population size (number of portfolios per generation).
    n_generations:
        Number of evolutionary generations.
    crossover_prob:
        Probability of applying crossover to a pair (vs. direct copy).
    mutation_prob:
        Per-individual probability of applying a swap mutation.
    seed:
        Random seed.
    hv_reference_point:
        Optional (3,) reference point for inline HV tracking.
    """

    def __init__(
        self,
        pop_size: int = 100,
        n_generations: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
        seed: int = 0,
        hv_reference_point: np.ndarray | None = None,
    ) -> None:
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed
        self.hv_reference_point = hv_reference_point

    def solve(self, problem: Any) -> OptimizationResult:

        N, K = problem.N, problem.K
        evaluator = problem.evaluator

        archive = ParetoArchive(problem, hv_cadence=10)
        if self.hv_reference_point is not None:
            archive.set_reference_point(self.hv_reference_point)

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # pymoo Problem definition
        # ------------------------------------------------------------------
        class _PortfolioProblem(Problem):
            def __init__(self_inner):
                super().__init__(n_var=N, n_obj=3, xl=0.0, xu=1.0)

            def _evaluate(self_inner, X, out, *args, **kwargs):
                # X: (pop_size, N) float, interpret >0.5 as selected
                X_bin = (X > 0.5).astype(np.float64)
                objs = evaluator.evaluate_batch(X_bin)
                out["F"] = -objs  # pymoo minimizes; negate for maximization

        # ------------------------------------------------------------------
        # Sampling: generate feasible K-portfolios
        # ------------------------------------------------------------------
        class _FeasibleSampling(Sampling):
            def _do(self_inner, problem_inner, n_samples, **kwargs):
                rng = np.random.default_rng(self.seed)
                X = np.zeros((n_samples, N), dtype=np.float64)
                for b in range(n_samples):
                    idx = rng.choice(N, size=K, replace=False)
                    X[b, idx] = 1.0
                return X

        # ------------------------------------------------------------------
        # Crossover: union-based, preserves cardinality K
        # ------------------------------------------------------------------
        class _UnionCrossover(Crossover):
            def __init__(self_inner):
                super().__init__(2, 2)  # n_parents=2, n_offsprings=2

            def _do(self_inner, problem_inner, X, **kwargs):
                # X shape: (n_parents=2, n_matings, n_var)
                _, n_matings, _ = X.shape
                off = np.zeros_like(X)
                rng = np.random.default_rng(self.seed)

                for m in range(n_matings):
                    p1 = X[0, m]
                    p2 = X[1, m]
                    if rng.random() < self.crossover_prob:
                        off[0, m] = _union_crossover(p1, p2, K, N, rng)
                        off[1, m] = _union_crossover(p2, p1, K, N, rng)
                    else:
                        off[0, m] = p1.copy()
                        off[1, m] = p2.copy()
                return off

        # ------------------------------------------------------------------
        # Mutation: swap one in ↔ one out
        # ------------------------------------------------------------------
        class _SwapMutation(Mutation):
            def __init__(self_inner):
                super().__init__()

            def _do(self_inner, problem_inner, X, **kwargs):
                rng = np.random.default_rng(self.seed)
                out = X.copy()
                for i in range(len(X)):
                    if rng.random() < self.mutation_prob:
                        x = out[i].copy()
                        selected = np.where(x > 0.5)[0]
                        unselected = np.where(x <= 0.5)[0]
                        if len(selected) > 0 and len(unselected) > 0:
                            remove = rng.choice(selected)
                            add = rng.choice(unselected)
                            x[remove] = 0.0
                            x[add] = 1.0
                        out[i] = x
                return out

        # ------------------------------------------------------------------
        # Callback: feed each generation into the external archive
        # ------------------------------------------------------------------
        gen_bar = tqdm(total=self.n_generations, desc="NSGA-II generations", unit="gen", leave=False)

        class _ArchiveCallback(Callback):
            def notify(self_inner, alg):
                pop = alg.pop
                X = pop.get("X")
                F = pop.get("F")  # negated objectives from pymoo
                if X is None or F is None:
                    return
                objs = -F  # re-negate to get maximization objectives
                X_bin = (X > 0.5).astype(np.float64)
                elapsed = time.perf_counter() - t_start
                archive.add_batch(X_bin, objs, elapsed_time=elapsed, source=f"nsga2_gen{alg.n_gen}")
                gen_bar.update(1)
                gen_bar.set_postfix(nd=archive.n_nondominated)

        # ------------------------------------------------------------------
        # Run NSGA-II
        # ------------------------------------------------------------------
        pym_problem = _PortfolioProblem()
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=_FeasibleSampling(),
            crossover=_UnionCrossover(),
            mutation=_SwapMutation(),
        )

        try:
            minimize(
                pym_problem,
                algorithm,
                ("n_gen", self.n_generations),
                seed=self.seed,
                callback=_ArchiveCallback(),
                verbose=False,
            )
        except Exception as exc:
            _log.warning("NSGA-II optimization raised exception: %s", exc)
        finally:
            gen_bar.close()

        runtime = time.perf_counter() - t_start
        n_evaluations = self.pop_size * self.n_generations
        status = "feasible" if archive.n_nondominated > 0 else "failed"

        _log.info(
            "NSGA-II: %d gen × %d pop, %d non-dominated, t=%.2fs",
            self.n_generations,
            self.pop_size,
            archive.n_nondominated,
            runtime,
        )

        return self._make_result(
            algorithm="nsga2",
            problem=problem,
            archive=archive,
            n_evaluations=n_evaluations,
            runtime=runtime,
            status=status,
            classical={
                "pop_size": self.pop_size,
                "n_generations": self.n_generations,
                "crossover_prob": self.crossover_prob,
                "mutation_prob": self.mutation_prob,
                "seed": self.seed,
            },
        )


def _union_crossover(
    p1: np.ndarray, p2: np.ndarray, K: int, N: int, rng: np.random.Generator
) -> np.ndarray:
    """Union crossover: select K stocks from parents' union.

    Takes the union of both parents' selected stocks; randomly draws K
    from it; fills remaining slots from unselected stocks if union < K.
    """
    sel1 = set(int(i) for i in np.where(p1 > 0.5)[0])
    sel2 = set(int(i) for i in np.where(p2 > 0.5)[0])
    union = list(sel1 | sel2)

    child = np.zeros(N, dtype=np.float64)
    if len(union) >= K:
        chosen = rng.choice(union, size=K, replace=False)
    else:
        # union is smaller than K — add random stocks from the complement
        complement = [i for i in range(N) if i not in (sel1 | sel2)]
        needed = K - len(union)
        if len(complement) >= needed:
            extra = rng.choice(complement, size=needed, replace=False)
        else:
            extra = np.array(complement)
            # Still not enough — should not happen for valid parents, but guard anyway
            still_needed = needed - len(complement)
            if still_needed > 0:
                extra = np.array(list(extra) + list(rng.choice(N, size=still_needed, replace=False)))
        chosen = np.array(union + list(extra))
        chosen = chosen[:K]

    child[chosen] = 1.0
    return child
