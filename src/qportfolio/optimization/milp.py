"""Binary Quadratic Program linearizer for portfolio optimization.

Converts the cardinality-constrained multi-objective portfolio problem into a
Mixed-Integer Linear Program (MILP) solved by HiGHS via scipy.optimize.milp.

Linearization:
  x^T Q x  →  sum_i Q_ii x_i + 2 * sum_{i<j} Q_ij y_ij

where y_ij = x_i * x_j is enforced by:
  y_ij ≤ x_i
  y_ij ≤ x_j
  y_ij ≥ x_i + x_j - 1

Variables: [x_0, ..., x_{N-1}, y_01, y_02, ..., y_{N-2,N-1}]
            N binary              N*(N-1)/2 continuous in [0,1]
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.sparse import vstack as sp_vstack

from qportfolio.utils.logging import get_logger

_log = get_logger("optimization.milp")


@dataclass
class MILPResult:
    """Result of a single MILP solve."""

    x: np.ndarray | None  # shape (N,) binary solution, or None if infeasible
    objectives: tuple[float, float, float] | None  # (f1, f2, f3) at solution
    status: Literal["optimal", "feasible", "infeasible", "timeout", "error"]
    solver_status: int  # raw scipy milp result.status
    mip_gap: float | None  # relative MIP gap at termination
    runtime: float  # wall-clock seconds
    message: str


class PortfolioMILP:
    """Builds and solves MILP instances for the portfolio optimization problem.

    Parameters
    ----------
    mu:
        Shape (N,) annualized expected returns.
    sigma:
        Shape (N, N) covariance matrix (used for f1 and optional ε constraints).
    D:
        Shape (N, N) diversification matrix.
    K:
        Number of stocks to select.
    """

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        D: np.ndarray,
        K: int,
    ) -> None:
        self._mu = np.asarray(mu, dtype=np.float64)
        self._sigma = np.asarray(sigma, dtype=np.float64)
        self._D = np.asarray(D, dtype=np.float64)
        self._K = K
        self._N = len(mu)

        # Enumerate all pairs (i, j) with i < j once
        self._pairs: list[tuple[int, int]] = [
            (i, j) for i in range(self._N) for j in range(i + 1, self._N)
        ]
        self._n_y = len(self._pairs)
        self._n_var = self._N + self._n_y
        self._pair_idx: dict[tuple[int, int], int] = {p: k for k, p in enumerate(self._pairs)}

        # Precompute diagonal and pair-indexed values for vectorized objective construction
        self._sigma_diag = np.diag(self._sigma)  # shape (N,)
        self._D_diag = np.diag(self._D)  # shape (N,)
        _ii = np.array([i for (i, j) in self._pairs], dtype=np.intp)
        _jj = np.array([j for (i, j) in self._pairs], dtype=np.intp)
        self._sigma_pair_vals = 2.0 * self._sigma[_ii, _jj]  # shape (n_y,)
        self._D_pair_vals = 2.0 * self._D[_ii, _jj]  # shape (n_y,)

        # Pre-build the fixed linearization constraints (cardinality + y_ij)
        self._lin_A, self._lin_lb, self._lin_ub = self._build_linearization_constraints()
        # Pre-build integrality and variable bounds
        self._integrality, self._var_bounds = self._build_variable_specs()

        _log.debug(
            "PortfolioMILP: N=%d, K=%d, n_vars=%d (%d binary + %d auxiliary)",
            self._N,
            self._K,
            self._n_var,
            self._N,
            self._n_y,
        )

    # ------------------------------------------------------------------
    # Public solve methods
    # ------------------------------------------------------------------

    def solve_wsm(
        self,
        weights: np.ndarray,
        time_limit: float | None = None,
        mip_rel_gap: float = 1e-4,
    ) -> MILPResult:
        """Solve a single Weighted Sum Method scalarization.

        Minimizes:  c1 * x^T Σ x - c2 * μ^T x - c3 * x^T D x
        Subject to: sum(x_i) = K,  x_i ∈ {0,1}

        Parameters
        ----------
        weights:
            Shape (3,) weight vector (c1, c2, c3) summing to 1.
        time_limit:
            Wall-clock time limit in seconds (None = no limit).
        mip_rel_gap:
            Relative MIP gap tolerance.

        Returns
        -------
        MILPResult
        """
        c_obj = self._build_wsm_objective(weights)
        return self._solve(
            c_obj, self._lin_A, self._lin_lb, self._lin_ub, time_limit, mip_rel_gap, label="wsm"
        )

    def solve_epsilon_cm(
        self,
        weights: np.ndarray,
        epsilons: np.ndarray,
        time_limit: float | None = None,
        mip_rel_gap: float = 1e-4,
    ) -> MILPResult:
        """Solve a randomized ε-constraint scalarization.

        Maximizes:    sum_j c_j f_j(x)
        Subject to:   f_j(x) ≥ ε_j  for j=1,2,3
                      sum(x_i) = K,  x_i ∈ {0,1}

        Parameters
        ----------
        weights:
            Shape (3,) weight vector.
        epsilons:
            Shape (3,) lower-bound thresholds for each objective.
        time_limit:
            Time limit in seconds.

        Returns
        -------
        MILPResult
        """
        c_obj = self._build_wsm_objective(weights)
        A_eps, lb_eps, ub_eps = self._build_epsilon_constraints(epsilons)

        # Stack linearization + epsilon constraints

        A_full = sp_vstack([self._lin_A, A_eps], format="csr")
        lb_full = np.concatenate([self._lin_lb, lb_eps])
        ub_full = np.concatenate([self._lin_ub, ub_eps])

        return self._solve(c_obj, A_full, lb_full, ub_full, time_limit, mip_rel_gap, label="epsilon_cm")

    # ------------------------------------------------------------------
    # Objective bounds (for ε-CM threshold sampling)
    # ------------------------------------------------------------------

    def compute_objective_bounds(
        self,
        time_limit_per_bound: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute lower and upper bounds for each objective using MILP.

        Returns
        -------
        (lower_bounds, upper_bounds) each of shape (3,)
        """
        lower = np.empty(3)
        upper = np.empty(3)

        # For each objective j: max and min using unit weight vector e_j
        for j in range(3):
            w = np.zeros(3)
            w[j] = 1.0

            # Maximize f_j (minimize -f_j)
            r_max = self.solve_wsm(w, time_limit=time_limit_per_bound)
            if r_max.objectives is not None:
                upper[j] = r_max.objectives[j]
            else:
                upper[j] = 0.0

            # Minimize f_j (maximize -f_j → weights with c_j = -1 doesn't make sense
            # for a maximization problem; instead solve with negated weight)
            # Minimize f_j = maximize -f_j, but the WSM objective is min c1*f1-c2*f2-c3*f3.
            # To minimize f_j, we maximize -f_j, which corresponds to negating only that weight.
            # Simplest: solve with weight that selects -f_j direction.
            w_neg = np.zeros(3)
            w_neg[j] = -1.0  # will be passed as-is to _build_wsm_objective
            r_min = self._solve_raw_objective(w_neg, time_limit_per_bound)
            if r_min.objectives is not None:
                lower[j] = r_min.objectives[j]
            else:
                lower[j] = 0.0

        _log.info("Objective bounds: lower=%s, upper=%s", lower.round(4), upper.round(4))
        return lower, upper

    def _solve_raw_objective(self, weights: np.ndarray, time_limit: float | None) -> MILPResult:
        """Like solve_wsm but allows negative weights (for bound computation)."""
        c_obj = self._build_wsm_objective(weights)
        return self._solve(
            c_obj, self._lin_A, self._lin_lb, self._lin_ub, time_limit, 1e-4, label="bound"
        )

    # ------------------------------------------------------------------
    # Internal: objective construction
    # ------------------------------------------------------------------

    def _build_wsm_objective(self, weights: np.ndarray) -> np.ndarray:
        """Build the linear objective vector for the scalarized portfolio problem.

        The objective to minimize is:
          c1 * x^T Σ x  - c2 * μ^T x  - c3 * x^T D x

        After linearization (x_i^2 = x_i, y_ij = x_i*x_j for i<j):
          = c1 * [sum_i Σ_ii*x_i + 2*sum_{i<j} Σ_ij*y_ij]
          - c2 * [sum_i μ_i*x_i]
          - c3 * [sum_i D_ii*x_i + 2*sum_{i<j} D_ij*y_ij]
        """
        c1, c2, c3 = weights[0], weights[1], weights[2]
        obj = np.empty(self._n_var)
        obj[: self._N] = c1 * self._sigma_diag - c3 * self._D_diag - c2 * self._mu
        obj[self._N :] = c1 * self._sigma_pair_vals - c3 * self._D_pair_vals
        return obj

    # ------------------------------------------------------------------
    # Internal: constraint construction
    # ------------------------------------------------------------------

    def _build_linearization_constraints(
        self,
    ) -> tuple[object, np.ndarray, np.ndarray]:
        """Build the y_ij = x_i*x_j linearization + cardinality constraint.

        Returns (A_sparse, lb, ub) for scipy LinearConstraint.
        """

        n_pairs = self._n_y
        # 3 constraints per pair + 1 cardinality
        n_rows = 3 * n_pairs + 1
        A = lil_matrix((n_rows, self._n_var))
        lb = np.empty(n_rows)
        ub = np.empty(n_rows)

        for k, (i, j) in enumerate(self._pairs):
            y_col = self._N + k
            row_base = 3 * k

            # y_ij ≤ x_i  →  -x_i + y_ij ≤ 0  →  lb=-inf, ub=0
            A[row_base, i] = -1.0
            A[row_base, y_col] = 1.0
            lb[row_base] = -np.inf
            ub[row_base] = 0.0

            # y_ij ≤ x_j  →  -x_j + y_ij ≤ 0
            A[row_base + 1, j] = -1.0
            A[row_base + 1, y_col] = 1.0
            lb[row_base + 1] = -np.inf
            ub[row_base + 1] = 0.0

            # y_ij ≥ x_i + x_j - 1  →  x_i + x_j - y_ij ≤ 1
            A[row_base + 2, i] = 1.0
            A[row_base + 2, j] = 1.0
            A[row_base + 2, y_col] = -1.0
            lb[row_base + 2] = -np.inf
            ub[row_base + 2] = 1.0

        # Cardinality constraint: sum_i x_i = K
        card_row = 3 * n_pairs
        for i in range(self._N):
            A[card_row, i] = 1.0
        lb[card_row] = float(self._K)
        ub[card_row] = float(self._K)

        return A.tocsr(), lb, ub

    def _build_epsilon_constraints(
        self,
        epsilons: np.ndarray,
    ) -> tuple[object, np.ndarray, np.ndarray]:
        """Build ε-constraint rows: f_j(x) ≥ ε_j for j=0,1,2.

        Linearized:
          f1(x) ≥ ε1:  -(Σ_ii*x_i + 2*Σ_ij*y_ij) ≥ ε1
                       → sum_i Σ_ii*x_i + 2*sum_{i<j} Σ_ij*y_ij ≤ -ε1
          f2(x) ≥ ε2:  μ^T x ≥ ε2 (linear)
          f3(x) ≥ ε3:  x^T D x ≥ ε3
                       → -(sum_i D_ii*x_i + 2*sum_{i<j} D_ij*y_ij) ≤ -ε3
        """

        n_rows = 3
        A = lil_matrix((n_rows, self._n_var))
        lb = np.empty(n_rows)
        ub = np.empty(n_rows)

        # Row 0: f1 ≥ ε1  →  Σ_ii*x_i + 2*Σ_ij*y_ij ≤ -ε1
        A[0, : self._N] = self._sigma_diag
        A[0, self._N :] = self._sigma_pair_vals
        lb[0] = -np.inf
        ub[0] = -float(epsilons[0])

        # Row 1: f2 ≥ ε2  →  μ^T x ≥ ε2
        A[1, : self._N] = self._mu
        lb[1] = float(epsilons[1])
        ub[1] = np.inf

        # Row 2: f3 ≥ ε3  →  -(D_ii*x_i + 2*D_ij*y_ij) ≤ -ε3
        A[2, : self._N] = -self._D_diag
        A[2, self._N :] = -self._D_pair_vals
        lb[2] = -np.inf
        ub[2] = -float(epsilons[2])

        return A.tocsr(), lb, ub

    def _build_variable_specs(
        self,
    ) -> tuple[np.ndarray, object]:
        """Build integrality flags and variable bounds."""
        from scipy.optimize import Bounds

        integrality = np.zeros(self._n_var)
        integrality[: self._N] = 1  # x variables are binary (integer + bounds [0,1])

        lb_var = np.zeros(self._n_var)
        ub_var = np.ones(self._n_var)

        return integrality, Bounds(lb=lb_var, ub=ub_var)

    # ------------------------------------------------------------------
    # Internal: solver invocation
    # ------------------------------------------------------------------

    def _solve(
        self,
        c_obj: np.ndarray,
        A: object,
        lb: np.ndarray,
        ub: np.ndarray,
        time_limit: float | None,
        mip_rel_gap: float,
        label: str,
    ) -> MILPResult:
        """Invoke scipy.optimize.milp and decode the result."""

        constraints = LinearConstraint(A, lb, ub)
        options: dict = {"mip_rel_gap": mip_rel_gap, "disp": False}
        if time_limit is not None:
            options["time_limit"] = time_limit

        t0 = _time.perf_counter()
        try:
            result = milp(
                c=c_obj,
                constraints=constraints,
                integrality=self._integrality,
                bounds=self._var_bounds,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001
            runtime = _time.perf_counter() - t0
            _log.warning("MILP (%s) raised exception: %s", label, exc)
            return MILPResult(None, None, "error", -1, None, runtime, str(exc))
        runtime = _time.perf_counter() - t0

        # scipy milp statuses: 0=optimal, 1=iteration/time limit with feasible,
        #                       2=infeasible, 3=unbounded, 4=other
        status_map: dict[int, Literal["optimal", "feasible", "infeasible", "timeout", "error"]] = {
            0: "optimal",
            1: "feasible",
            2: "infeasible",
            3: "infeasible",
        }
        status = status_map.get(result.status, "error")

        if result.x is None or status == "infeasible":
            _log.debug("MILP (%s) infeasible or failed (status=%d)", label, result.status)
            return MILPResult(None, None, status, result.status, None, runtime, result.message)

        x_sol = np.round(result.x[: self._N]).astype(np.float64)
        # Decode objectives from the *original* objective functions
        f1 = float(-x_sol @ self._sigma @ x_sol)
        f2 = float(self._mu @ x_sol)
        f3 = float(x_sol @ self._D @ x_sol)

        mip_gap = getattr(result, "mip_gap", None)
        _log.debug(
            "MILP (%s) %s: f1=%.4f, f2=%.4f, f3=%.4f, t=%.2fs",
            label,
            status,
            f1,
            f2,
            f3,
            runtime,
        )
        return MILPResult(x_sol, (f1, f2, f3), status, result.status, mip_gap, runtime, result.message)
