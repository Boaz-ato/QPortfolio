"""Example 5: Benchmark multiple solvers on a portfolio problem.

Runs a BenchmarkSuite comparing all solvers on a saved or freshly generated
problem, then prints a structured comparison table with hypervolume,
non-dominated count, and runtime.

Three stopping modes are supported:
  - fixed_time        : each solver gets a wall-clock budget (seconds)
  - fixed_evaluations : each solver runs until N unique feasible portfolios found
  - fixed_samples     : each solver generates N raw candidates

Usage:
    # Full benchmark (all solvers including QAOA)
    python examples/05_benchmark.py

    # Load a previously saved problem
    python examples/05_benchmark.py --problem-dir examples/saved_problems/<problem_id>

    # Fixed-evaluations mode with 3 repetitions per solver
    python examples/05_benchmark.py --mode fixed_evaluations --budget 500 --repeat 3

    # Tune QAOA parameters (runs p=1..3)
    python examples/05_benchmark.py --qaoa-p-max 3 --qaoa-n-weights 30 --qaoa-shots 512 --qaoa-iters 50

    # Classical solvers only
    python examples/05_benchmark.py --no-qaoa
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from tqdm import tqdm

from qportfolio import (
    EpsilonCMSolver,
    NSGA2Solver,
    PortfolioProblem,
    QAOASolver,
    RandomSampler,
)
from qportfolio.evaluation.benchmark import BenchmarkConfig, BenchmarkSuite


def build_default_problem() -> PortfolioProblem:
    """Synthetic N=8, K=3 problem — no network required."""
    rng = np.random.default_rng(42)
    N, K = 8, 3
    tickers = [f"T{i:02d}" for i in range(N)]
    sectors = ["Tech", "Health", "Finance", "Energy"] * 2
    sector_map = dict(zip(tickers, sectors))
    mu = rng.uniform(0.08, 0.30, N)
    A = rng.normal(0, 0.12, (N, N))
    sigma = (A @ A.T) / N + 0.02 * np.eye(N)
    return PortfolioProblem.from_arrays(
        tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K, seed=42
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark solvers on a portfolio problem")

    # Problem
    parser.add_argument("--problem-dir", default=None, help="Path to a saved PortfolioProblem directory")

    # Benchmark config
    parser.add_argument(
        "--mode",
        choices=["fixed_time", "fixed_evaluations", "fixed_samples"],
        default="fixed_time",
        help="Stopping criterion (default: fixed_time)",
    )
    parser.add_argument(
        "--budget", type=float, default=60.0 * 60, help="Budget: seconds / evaluations / samples"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repetitions per solver (seeds incremented)"
    )
    parser.add_argument(
        "--reference",
        choices=["exact", "conservative"],
        default="conservative",
        help="Reference point strategy for hypervolume (default: conservative)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for all solvers")

    # Classical solver args
    parser.add_argument(
        "--random-samples", type=int, default=1000000, help="Random sampler: number of samples"
    )
    parser.add_argument("--wsm-weights", type=int, default=100, help="WSM: number of weight vectors")
    parser.add_argument("--epsilon-samples", type=int, default=100, help="Epsilon-CM: number of samples")
    parser.add_argument(
        "--nsga2-generations", type=int, default=50, help="NSGA-II: number of generations"
    )
    parser.add_argument("--nsga2-pop-size", type=int, default=100, help="NSGA-II: population size")

    # QAOA args
    parser.add_argument("--no-qaoa", action="store_true", help="Skip QAOA solvers")
    parser.add_argument("--qaoa-p-max", type=int, default=4, help="QAOA: run p=1..p-max (default: 6)")
    parser.add_argument(
        "--qaoa-n-weights", type=int, default=30, help="QAOA: number of weight vectors (default: 20)"
    )
    parser.add_argument(
        "--qaoa-shots", type=int, default=512, help="QAOA: shots per weight vector (default: 256)"
    )
    parser.add_argument(
        "--qaoa-iters", type=int, default=30, help="QAOA: training iterations (default: 30)"
    )

    args = parser.parse_args()

    # ── Problem ───────────────────────────────────────────────────────────────
    if args.problem_dir:
        print(f"Loading problem from {args.problem_dir}...")
        problem = PortfolioProblem.load(args.problem_dir)
    else:
        print("Building synthetic problem (N=8, K=3)...")
        problem = build_default_problem()

    print(f"  ID      : {problem.problem_id}")
    print(f"  Tickers : {problem.tickers}")
    print(f"  N={problem.N}, K={problem.K}")
    n_feasible = math.comb(problem.N, problem.K)
    print(f"  Feasible portfolios: C({problem.N},{problem.K}) = {n_feasible:,}")
    print()

    # ── Solvers ───────────────────────────────────────────────────────────────
    solvers = [
        RandomSampler(n_samples=args.random_samples, seed=args.seed),
        # WSMSolver(n_weights=args.wsm_weights, seed=args.seed),
        EpsilonCMSolver(n_samples=args.epsilon_samples, seed=args.seed),
        NSGA2Solver(n_generations=args.nsga2_generations, pop_size=args.nsga2_pop_size, seed=args.seed),
    ]

    if not args.no_qaoa:
        for p in range(1, args.qaoa_p_max + 1):
            solver = QAOASolver(
                p=p,
                n_weights=args.qaoa_n_weights,
                shots=args.qaoa_shots,
                n_training_iters=args.qaoa_iters,
                seed=args.seed,
                bond_dimension=32,
            )
            solver.algorithm_name = f"qaoa_p{p}"
            solvers.append(solver)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"Running benchmark: mode={args.mode}, budget={args.budget}, repeat={args.repeat}")
    print(f"Reference point strategy: {args.reference}")
    print("-" * 60)

    config = BenchmarkConfig(
        mode=args.mode,
        budget=args.budget,
        repeat=args.repeat,
        reference_strategy=args.reference,
        compute_normalized_hv=True,
    )
    suite = BenchmarkSuite(problem=problem, solvers=solvers, config=config)

    results = []
    n_solvers = len(suite.solvers)
    for i, solver in enumerate(suite.solvers):
        algo = getattr(solver, "algorithm_name", type(solver).__name__)
        with tqdm(total=1, desc=f"[{i + 1}/{n_solvers}] {algo}", unit="solver", leave=True) as bar:
            bench = suite._run_solver(
                solver, suite._ensure_reference_point(), suite._ensure_normalizer()
            )
            results.append(bench)
            hv_str = f"{bench.hypervolume:.4f}" if bench.hypervolume is not None else "N/A"
            bar.set_postfix(nd=bench.n_nondominated, hv=hv_str, t=f"{bench.runtime:.2f}s")
            bar.update(1)

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    print("=== Benchmark Results ===")
    table = suite.comparison_table(results)
    print(table.to_string())

    print()
    print(f"Reference point: {suite._reference_point}")

    # ── Save plots and data ───────────────────────────────────────────────────
    run_dir = problem.save_benchmark_results(results, output_dir="final_results")
    print(f"\nAll plots and results saved to: {run_dir}")


if __name__ == "__main__":
    main()
