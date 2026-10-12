"""Portfolio feasibility validation.

A portfolio x ∈ {0,1}^N is feasible iff:
  - x is binary (all values in {0, 1})
  - sum(x) == K   (exactly K stocks selected)
"""

from __future__ import annotations

import itertools

import numpy as np

from qportfolio.utils.logging import get_logger

_log = get_logger("problems.constraints")


class ConstraintValidator:
    """Validates portfolio feasibility for a fixed (N, K) problem.

    Parameters
    ----------
    N:
        Number of available stocks.
    K:
        Number of stocks to select.
    """

    def __init__(self, N: int, K: int) -> None:
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}.")
        if K < 1 or K > N:
            raise ValueError(f"K must be in [1, N]=[1, {N}], got {K}.")
        self.N = N
        self.K = K

    # ------------------------------------------------------------------
    # Feasibility checks
    # ------------------------------------------------------------------

    def is_feasible(self, x: np.ndarray) -> bool:
        """Return True iff x is binary and sums to K."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.N,):
            return False
        if not np.all((x == 0) | (x == 1)):
            return False
        return int(x.sum()) == self.K

    def is_feasible_batch(self, X: np.ndarray) -> np.ndarray:
        """Return boolean array of shape (M,) for a batch of portfolios.

        Parameters
        ----------
        X:
            Shape (M, N) array.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.N:
            raise ValueError(f"Expected shape (M, {self.N}), got {X.shape}.")
        is_binary = np.all((X == 0) | (X == 1), axis=1)
        has_k = X.sum(axis=1) == self.K
        return is_binary & has_k

    def feasibility_report(self, x: np.ndarray) -> dict:
        """Return a dict describing all feasibility violations.

        Returns
        -------
        dict with keys:
            is_binary, cardinality, K, is_feasible, violations (list[str])
        """
        x = np.asarray(x, dtype=float)
        violations: list[str] = []

        is_binary = bool(np.all((x == 0) | (x == 1)))
        if not is_binary:
            n_frac = int(np.sum((x != 0) & (x != 1)))
            violations.append(f"{n_frac} non-binary value(s) in x")

        cardinality = int(x.sum())
        if cardinality != self.K:
            violations.append(f"cardinality={cardinality} != K={self.K}")

        return {
            "is_binary": is_binary,
            "cardinality": cardinality,
            "K": self.K,
            "is_feasible": len(violations) == 0,
            "violations": violations,
        }

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def all_feasible_portfolios(self) -> np.ndarray:
        """Return all C(N, K) feasible portfolios as a (C(N,K), N) array.

        Only safe for small N. Raises OverflowError if C(N,K) > 1_000_000.

        Returns
        -------
        np.ndarray
            Shape (C(N,K), N) uint8 array. Each row is a binary portfolio.
        """
        from math import comb

        n_portfolios = comb(self.N, self.K)
        if n_portfolios > 1_000_000:
            raise OverflowError(
                f"C({self.N},{self.K}) = {n_portfolios:,} exceeds the safety limit of 1,000,000. "
                "Use a sampling method instead."
            )

        result = np.zeros((n_portfolios, self.N), dtype=np.uint8)
        for idx, combo in enumerate(itertools.combinations(range(self.N), self.K)):
            result[idx, list(combo)] = 1
        return result
