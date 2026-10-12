"""Example 1: Classical solver comparison on a synthetic portfolio problem.

Runs all four classical solvers (Random, WSM, ε-CM, NSGA-II) on a small
N=8, K=3 problem built from synthetic data, then prints a comparison table.

Usage:
    python examples/01_classical_comparison.py
"""

import numpy as np

from qportfolio import (
    EpsilonCMSolver,
    NSGA2Solver,
    PortfolioProblem,
    RandomSampler,
    WSMSolver,
)


def build_problem(seed: int = 42) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    N, K = 8, 3
    tickers = [f"T{i:02d}" for i in range(N)]
    sectors = ["Tech", "Health", "Finance", "Energy", "Tech", "Health", "Finance", "Energy"]
    sector_map = dict(zip(tickers, sectors))

    mu = rng.uniform(0.08, 0.30, N)
    A = rng.normal(0, 0.12, (N, N))
    sigma = (A @ A.T) / N + 0.02 * np.eye(N)

    return PortfolioProblem.from_arrays(tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K)


def main() -> None:
    problem = build_problem()

    print(f"Problem: N={problem.N}, K={problem.K}, id={problem.problem_id}")
    print(
        f"Feasible portfolios: C({problem.N},{problem.K}) = {len(problem.all_feasible_portfolios())}\n"
    )

    solvers = {
        "random": RandomSampler(n_samples=2000, seed=0),
        "wsm": WSMSolver(n_weights=50, seed=0),
        "epsilon": EpsilonCMSolver(n_samples=50, seed=0),
        "nsga2": NSGA2Solver(n_generations=50, pop_size=20, seed=0),
    }

    results = []
    for name, solver in solvers.items():
        print(f"Running {name}...")
        result = solver.solve(problem)
        results.append(result)
        s = result.summary()
        print(
            f"  {name:8s}  nd={s['n_nondominated']:3d}  "
            f"hv={s['hypervolume'] or 'N/A'!s:8}  t={s['runtime_s']:.3f}s"
        )

    print("\n=== Comparison Table ===")
    comparison = problem.compare(results)
    print(comparison.to_string())


if __name__ == "__main__":
    main()
