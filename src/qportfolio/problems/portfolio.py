"""PortfolioProblem — the central object of qportfolio.

Stores the optimization instance (μ, Σ, D, tickers, K) together with metadata
and provides factory methods for construction, save/load, and objective evaluation.

The problem_id is a deterministic sha256-based hash of the canonical problem data,
invariant to ticker insertion order. It is used to validate serialization integrity
and to reject cross-problem comparisons.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast

import numpy as np

from qportfolio.config import CovarianceConfig, DataConfig, UniverseConfig
from qportfolio.data.cache import DiskCache
from qportfolio.data.covariance import CovarianceEstimator
from qportfolio.data.preprocessing import (
    annualize_returns,
    build_diversification_matrix,
    compute_daily_returns,
)
from qportfolio.data.universe import StockUniverse
from qportfolio.data.yahoo import MarketDataProvider
from qportfolio.evaluation.hypervolume import _pymoo_hv
from qportfolio.evaluation.reference import (
    ReferenceStrategy,
    build_normalizer,
    compute_reference_point,
)
from qportfolio.problems.constraints import ConstraintValidator
from qportfolio.problems.objectives import ObjectiveEvaluator
from qportfolio.utils.logging import get_logger
from qportfolio.utils.validation import (
    IntegrityError,
    validate_date_range,
    validate_shape_consistency,
    validate_ticker_list,
)

_log = get_logger("problems.portfolio")

_SCHEMA_VERSION = "1.0"


@dataclass
class EvaluationContext:
    """Immutable settings used consistently across all solvers for this problem."""

    reference_point: np.ndarray | None = None
    objective_scaling: np.ndarray | None = None  # shape (3,) normalization denominators
    feasibility_tolerance: float = 1e-8


@dataclass
class PortfolioProblem:
    """Multi-objective portfolio optimization problem instance.

    Use the factory methods `from_yfinance` or `from_arrays` to construct instances.

    Attributes
    ----------
    tickers: list[str]
        Canonical ordered list of N stock symbols.
    mu: np.ndarray
        Shape (N,) annualized expected returns.
    sigma: np.ndarray
        Shape (N, N) full annualized covariance matrix.
    D: np.ndarray
        Shape (N, N) sector diversification matrix.
    K: int
        Number of stocks to select.
    sector_map: dict[str, str]
        Mapping from ticker to sector name.
    sigma_sparse: np.ndarray | None
        Optional sparse covariance (for quantum circuit construction).
    sparsification_report: dict | None
        Metadata from covariance sparsification.
    start_date, end_date: str
        Requested date range.
    actual_start, actual_end: str
        Actual first/last trading dates returned by the provider.
    n_trading_days: int
    universe_mode: str
    universe_registry_version: str
    seed: int | None
    created_at: str
        ISO 8601 UTC timestamp.
    problem_id: str
        16-character hex identifier derived from the problem data.
    evaluation: EvaluationContext
        Shared evaluation settings (reference point, scaling).
    """

    # Required
    tickers: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    D: np.ndarray
    K: int
    sector_map: dict[str, str]

    # Optional sparse covariance
    sigma_sparse: np.ndarray | None = None
    sparsification_report: dict | None = None

    # Metadata
    start_date: str = ""
    end_date: str = ""
    actual_start: str = ""
    actual_end: str = ""
    n_trading_days: int = 0
    universe_mode: str = "sector_balanced"
    universe_registry_version: str = "v1"
    seed: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Derived (set in __post_init__)
    problem_id: str = field(init=False, default="")
    evaluation: EvaluationContext = field(default_factory=EvaluationContext)

    # Lazy-initialized private caches (excluded from dataclass equality / repr)
    _evaluator: ObjectiveEvaluator | None = field(default=None, init=False, repr=False, compare=False)
    _validator: ConstraintValidator | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_ticker_list(self.tickers)
        validate_shape_consistency(self.tickers, self.mu, self.sigma, self.D)
        self.problem_id = self._compute_problem_id()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        """Number of available stocks."""
        return len(self.tickers)

    @property
    def evaluator(self) -> ObjectiveEvaluator:
        """Lazy-initialized ObjectiveEvaluator using the *full* covariance."""
        if self._evaluator is None:
            self._evaluator = ObjectiveEvaluator(self.mu, self.sigma, self.D)
        return self._evaluator

    @property
    def validator(self) -> ConstraintValidator:
        """Lazy-initialized ConstraintValidator."""
        if self._validator is None:
            self._validator = ConstraintValidator(self.N, self.K)
        return self._validator

    # ------------------------------------------------------------------
    # Objective evaluation (convenience delegation)
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> tuple[float, float, float]:
        """Evaluate (f1, f2, f3) for a single binary portfolio vector."""
        return self.evaluator.evaluate(x)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate (f1, f2, f3) for a batch of portfolios. Returns shape (M, 3)."""
        return self.evaluator.evaluate_batch(X)

    def all_feasible_portfolios(self) -> np.ndarray:
        """Return all C(N,K) feasible portfolios. Safe only for small N."""
        return self.validator.all_feasible_portfolios()

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict."""
        sectors_used = sorted(set(self.sector_map.values()))
        return {
            "problem_id": self.problem_id,
            "N": self.N,
            "K": self.K,
            "tickers": self.tickers,
            "sectors": sectors_used,
            "date_range": f"{self.actual_start or self.start_date} → {self.actual_end or self.end_date}",
            "n_trading_days": self.n_trading_days,
            "has_sparse_covariance": self.sigma_sparse is not None,
            "universe_mode": self.universe_mode,
            "seed": self.seed,
            "created_at": self.created_at,
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yfinance(
        cls,
        n_stocks: int,
        n_select: int,
        start: str,
        end: str,
        seed: int = 42,
        universe_mode: str = "sector_balanced",
        custom_tickers: list[str] | None = None,
        covariance: str = "full",
        sparsification: dict | None = None,
        cache_dir: str | None = None,
        data_config: Any | None = None,
    ) -> PortfolioProblem:
        """Build a PortfolioProblem by downloading data from Yahoo Finance.

        Parameters
        ----------
        n_stocks:
            Number of stocks in the universe.
        n_select (K):
            Number of stocks to select in each portfolio.
        start, end:
            Requested date range (YYYY-MM-DD).
        seed:
            Random seed for universe selection.
        universe_mode:
            'sector_balanced', 'random', or 'custom'.
        custom_tickers:
            Required when universe_mode='custom'.
        covariance:
            'full' or 'sparse'.
        sparsification:
            Dict of CovarianceConfig kwargs (threshold, max_risk_error, etc.).
        cache_dir:
            Override default cache directory.
        data_config:
            Optional DataConfig instance.
        """

        validate_date_range(start, end)

        # --- Universe selection ---
        u_config = UniverseConfig(
            mode=cast(Literal["sector_balanced", "random", "custom"], universe_mode),
            n_stocks=n_stocks,
            seed=seed,
            custom_tickers=custom_tickers or [],
        )
        universe = StockUniverse(u_config)
        tickers = universe.select()
        _log.info("Selected %d tickers (mode=%s)", len(tickers), universe_mode)

        # --- Data download ---
        d_config = data_config or DataConfig(start=start, end=end)
        cache = DiskCache(cache_dir) if cache_dir else DiskCache(d_config.cache_dir)
        provider = MarketDataProvider(d_config, cache)

        prices = provider.fetch_prices(tickers, start, end)
        sector_map = provider.fetch_sector_map(tickers)

        actual_start = str(prices.index[0].date())
        actual_end = str(prices.index[-1].date())
        n_trading_days = len(prices)

        # --- Financial computations ---
        daily_returns = compute_daily_returns(prices)
        mu = annualize_returns(daily_returns)
        D = build_diversification_matrix(tickers, sector_map)

        # --- Covariance estimation ---
        cov_method = "sparse" if covariance == "sparse" else "full"
        cov_kwargs = sparsification or {}
        cov_config = CovarianceConfig(method=cast(Literal["full", "sparse"], cov_method), **cov_kwargs)
        estimator = CovarianceEstimator(cov_config)
        estimator.fit(daily_returns)

        return cls(
            tickers=tickers,
            mu=mu,
            sigma=estimator.sigma_full_,
            D=D,
            K=n_select,
            sector_map=sector_map,
            sigma_sparse=estimator.sigma_sparse_,
            sparsification_report=estimator.sparsification_report_,
            start_date=start,
            end_date=end,
            actual_start=actual_start,
            actual_end=actual_end,
            n_trading_days=n_trading_days,
            universe_mode=universe_mode,
            universe_registry_version=universe.registry_version(),
            seed=seed,
        )

    @classmethod
    def from_arrays(
        cls,
        tickers: list[str],
        mu: np.ndarray,
        sigma: np.ndarray,
        sector_map: dict[str, str],
        K: int,
        D: np.ndarray | None = None,
        **metadata_kwargs: Any,
    ) -> PortfolioProblem:
        """Construct a PortfolioProblem from pre-computed arrays.

        Parameters
        ----------
        tickers:
            Ordered list of N ticker symbols.
        mu:
            Shape (N,) expected returns.
        sigma:
            Shape (N, N) covariance matrix.
        sector_map:
            {ticker: sector_name}.
        K:
            Number of stocks to select.
        D:
            Shape (N, N) diversification matrix. If None, computed from sector_map.
        **metadata_kwargs:
            Any PortfolioProblem field values (start_date, seed, etc.).
        """

        if D is None:
            D = build_diversification_matrix(tickers, sector_map)

        return cls(
            tickers=list(tickers),
            mu=np.asarray(mu, dtype=np.float64),
            sigma=np.asarray(sigma, dtype=np.float64),
            D=np.asarray(D, dtype=np.float64),
            K=K,
            sector_map=dict(sector_map),
            **metadata_kwargs,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save the problem to a directory.

        Creates:
          <path>/problem.json   — metadata, tickers, K, sector_map
          <path>/mu.npy         — shape (N,)
          <path>/sigma.npy      — shape (N, N) full covariance
          <path>/D.npy          — shape (N, N) diversification matrix
          <path>/sigma_sparse.npy  — optional, only if sparse covariance exists

        Parameters
        ----------
        path:
            Directory path (will be created if it doesn't exist).
        """
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.save(path / "mu.npy", self.mu)
        np.save(path / "sigma.npy", self.sigma)
        np.save(path / "D.npy", self.D)
        if self.sigma_sparse is not None:
            np.save(path / "sigma_sparse.npy", self.sigma_sparse)

        # Save metadata JSON
        meta: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "tickers": self.tickers,
            "K": self.K,
            "sector_map": self.sector_map,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "actual_start": self.actual_start,
            "actual_end": self.actual_end,
            "n_trading_days": self.n_trading_days,
            "universe_mode": self.universe_mode,
            "universe_registry_version": self.universe_registry_version,
            "seed": self.seed,
            "created_at": self.created_at,
            "has_sparse_covariance": self.sigma_sparse is not None,
            "sparsification_report": self.sparsification_report,
        }
        with open(path / "problem.json", "w") as f:
            json.dump(meta, f, indent=2)

        _log.info("Saved PortfolioProblem (id=%s) to %s", self.problem_id, path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> PortfolioProblem:
        """Load a PortfolioProblem from a saved directory.

        Raises
        ------
        IntegrityError
            If the recomputed problem_id doesn't match the stored one.
        FileNotFoundError
            If required files are missing.
        """
        path = pathlib.Path(path)

        with open(path / "problem.json") as f:
            meta = json.load(f)

        # Schema version check — raise early with a clear message
        stored_schema = meta.get("schema_version", "unknown")
        if stored_schema != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {stored_schema!r}. "
                f"This installation supports schema_version={_SCHEMA_VERSION!r}. "
                "Re-save the problem with the current version of qportfolio."
            )

        mu = np.load(path / "mu.npy")
        sigma = np.load(path / "sigma.npy")
        D = np.load(path / "D.npy")

        sigma_sparse = None
        if meta.get("has_sparse_covariance"):
            sparse_path = path / "sigma_sparse.npy"
            if sparse_path.exists():
                sigma_sparse = np.load(sparse_path)

        problem = cls(
            tickers=meta["tickers"],
            mu=mu,
            sigma=sigma,
            D=D,
            K=meta["K"],
            sector_map=meta["sector_map"],
            sigma_sparse=sigma_sparse,
            sparsification_report=meta.get("sparsification_report"),
            start_date=meta.get("start_date", ""),
            end_date=meta.get("end_date", ""),
            actual_start=meta.get("actual_start", ""),
            actual_end=meta.get("actual_end", ""),
            n_trading_days=meta.get("n_trading_days", 0),
            universe_mode=meta.get("universe_mode", "unknown"),
            universe_registry_version=meta.get("universe_registry_version", "unknown"),
            seed=meta.get("seed"),
            created_at=meta.get("created_at", ""),
        )

        stored_id = meta["problem_id"]
        if problem.problem_id != stored_id:
            raise IntegrityError(
                f"problem_id mismatch: stored={stored_id!r}, "
                f"recomputed={problem.problem_id!r}. "
                "The problem data may have been corrupted."
            )

        _log.info("Loaded PortfolioProblem (id=%s) from %s", problem.problem_id, path)
        return problem

    # ------------------------------------------------------------------
    # problem_id derivation
    # ------------------------------------------------------------------

    def _compute_problem_id(self) -> str:
        """Compute a deterministic 16-char hex id from the canonical problem data.

        Algorithm:
        1. Sort tickers lexicographically; apply same permutation to mu, sigma, D.
        2. Round floats to 8 decimal places.
        3. Serialize as compact JSON with sorted keys.
        4. Return sha256[:16].

        The canonical sort makes the id invariant to ticker insertion order.
        Rounding avoids spurious mismatches from float noise in the last ULP.
        """
        perm = [i for _, i in sorted((t, i) for i, t in enumerate(self.tickers))]
        sorted_tickers = [self.tickers[i] for i in perm]
        mu_c = self.mu[perm]
        sigma_c = self.sigma[np.ix_(perm, perm)]
        D_c = self.D[np.ix_(perm, perm)]

        payload = {
            "tickers": sorted_tickers,
            "K": self.K,
            "mu": [round(float(v), 8) for v in mu_c],
            "sigma": [[round(float(v), 8) for v in row] for row in sigma_c],
            "D": [[int(v) for v in row] for row in D_c],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Visualization API
    # ------------------------------------------------------------------

    def plot_hypervolume_vs_time(self, results: list, **kwargs):
        """Plot hypervolume vs elapsed time for each solver. Returns matplotlib Figure."""
        from qportfolio.visualization.plots import plot_hypervolume_vs_time

        return plot_hypervolume_vs_time(results, **kwargs)

    def plot_nondominated_vs_time(self, results: list, **kwargs):
        """Plot non-dominated solution count vs elapsed time. Returns matplotlib Figure."""
        from qportfolio.visualization.plots import plot_nondominated_vs_time

        return plot_nondominated_vs_time(results, **kwargs)

    def plot_pareto_front_3d(self, results: list, **kwargs):
        """3D scatter of non-dominated solutions per solver. Returns matplotlib Figure."""
        from qportfolio.visualization.plots import plot_pareto_front_3d

        return plot_pareto_front_3d(results, **kwargs)

    def plot_runtime(self, results: list, **kwargs):
        """Horizontal bar chart of total runtime per solver. Returns matplotlib Figure."""
        from qportfolio.visualization.plots import plot_runtime

        return plot_runtime(results, **kwargs)

    def save_benchmark_results(
        self,
        results: list,
        output_dir=None,
        run_name: str | None = None,
    ):
        """Save all plots and data to a uniquely named results directory. Returns Path."""
        from qportfolio.visualization.plots import save_benchmark_results

        return save_benchmark_results(results, problem=self, output_dir=output_dir, run_name=run_name)

    # ------------------------------------------------------------------
    # Comparison API
    # ------------------------------------------------------------------

    def compare(
        self,
        results: list,
        reference_point: np.ndarray | None = None,
        reference_strategy: str = "exact",
        compute_normalized_hv: bool = True,
        milp_time_limit: float = 30.0,
    ) -> ComparisonTable:  # type: ignore[name-defined]
        """Compare multiple OptimizationResult objects on this problem.

        Parameters
        ----------
        results : list of OptimizationResult (from any solver).
        reference_point : optional; if None, computed from problem.
        reference_strategy : 'exact', 'conservative', or 'manual'.
        compute_normalized_hv : whether to include normalized HV column.
        milp_time_limit : MILP time per bound (used only for large problems).

        Returns
        -------
        ComparisonTable with one row per solver result.
        """
        from qportfolio.evaluation.benchmark import ComparisonTable

        # Resolve reference point
        if reference_point is None:
            ref_pt = compute_reference_point(
                self,
                strategy=ReferenceStrategy(reference_strategy),
                milp_time_limit=milp_time_limit,
            )
        else:
            ref_pt = np.asarray(reference_point, dtype=float)

        normalizer = None
        if compute_normalized_hv:
            normalizer = build_normalizer(
                self,
                strategy=ReferenceStrategy(reference_strategy),
                milp_time_limit=milp_time_limit,
            )

        rows = []
        for result in results:
            hv = None
            norm_hv = None
            if result.n_nondominated > 0:
                nd_objs = np.array([s.objectives for s in result.nondominated_solutions])
                try:
                    hv = _pymoo_hv(nd_objs, ref_pt)
                except Exception:
                    hv = None
                if normalizer is not None:
                    try:
                        norm_hv = normalizer.compute(nd_objs, ref_pt)
                    except Exception:
                        norm_hv = None

            rows.append(
                {
                    "algorithm": result.algorithm,
                    "status": result.status,
                    "n_nondominated": result.n_nondominated,
                    "n_feasible": result.n_feasible,
                    "n_evaluations": result.n_evaluations,
                    "hypervolume": hv,
                    "normalized_hv": norm_hv,
                    "runtime_s": result.runtime,
                }
            )

        return ComparisonTable(
            problem_id=self.problem_id,
            mode="comparison",
            budget=float("nan"),
            reference_point=ref_pt,
            rows=rows,
        )
