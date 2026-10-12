"""Example 2: QAOA on a synthetic portfolio problem.

Runs QAOASolver (p=1) using the Qiskit Aer MPS simulator,
then compares the result against a random baseline.

Usage:
    pip install qportfolio[quantum]
    python examples/02_qaoa_example.py
"""

import numpy as np

from qportfolio import PortfolioProblem, QAOASolver, RandomSampler


def build_problem(seed: int = 0) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    N, K = 20, 4
    tickers = [f"S{i}" for i in range(N)]
    sectors = ["Tech", "Health", "Tech", "Health"]
    sector_map = dict(zip(tickers, sectors))
    mu = rng.uniform(0.08, 0.28, N)
    A = rng.normal(0, 0.05, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)
    return PortfolioProblem.from_arrays(tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K)


def main() -> None:
    problem = build_problem()
    print(f"Problem: N={problem.N}, K={problem.K}, id={problem.problem_id}")

    # --- QAOA solver ---
    print("\nRunning QAOA (p=1, 20 weight vectors, 256 shots)...")
    qaoa = QAOASolver(p=1, n_weights=20, shots=256, seed=0, n_training_iters=30)
    qaoa_result = qaoa.solve(problem)

    meta = qaoa_result.quantum
    print(f"  Trained γ: {[f'{g:.4f}' for g in meta['params_gammas']]}")
    print(f"  Trained β: {[f'{b:.4f}' for b in meta['params_betas']]}")
    print(f"  HW violations: {meta['hardware_violations']}")
    print(f"  Unique feasible: {meta['unique_feasible']}")
    print(f"  Non-dominated: {qaoa_result.n_nondominated}")
    print(f"  Runtime: {qaoa_result.runtime:.2f}s")

    # --- Comparison against random baseline ---
    print("\nRunning random baseline (2000 samples)...")
    rnd = RandomSampler(n_samples=2000, seed=1)
    rnd_result = rnd.solve(problem)

    print("\n=== Comparison ===")
    comparison = problem.compare([qaoa_result, rnd_result])
    print(comparison.to_string())

    # --- Check all QAOA solutions are feasible ---
    all_feasible = all(int(s.bitstring.sum()) == problem.K for s in qaoa_result.solutions)
    print(f"\nAll QAOA solutions feasible (HW=K): {all_feasible}")


if __name__ == "__main__":
    main()
