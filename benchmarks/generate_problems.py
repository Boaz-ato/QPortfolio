"""Generate and save reproducible benchmark problem instances.

Creates PortfolioProblem instances at three scales:
  - Small:  N=8,  K=3  (C(8,3)=56 feasible portfolios)
  - Medium: N=20, K=5  (C(20,5)=15504 feasible portfolios)
  - Large:  N=40, K=8  (C(40,8)≈76M  feasible portfolios, too large to enumerate)

Each instance is saved to benchmarks/problems/<name>/ using PortfolioProblem.save().
The saved instances are fully reproducible from the RNG seed embedded in problem.json.

Usage:
    python benchmarks/generate_problems.py [--output-dir benchmarks/problems]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import time

import numpy as np

from qportfolio import PortfolioProblem

PROBLEM_SPECS = [
    # (name, N, K, seed, sectors)
    (
        "N8K3_seed0",
        8,
        3,
        0,
        ["Tech", "Health", "Finance", "Energy"] * 2,
    ),
    (
        "N20K5_seed0",
        20,
        5,
        0,
        (["Tech", "Health", "Finance", "Energy", "Utilities"] * 4),
    ),
    (
        "N40K8_seed0",
        40,
        8,
        0,
        (
            [
                "Tech",
                "Health",
                "Finance",
                "Energy",
                "Utilities",
                "Materials",
                "Industrials",
                "ConsumerGoods",
            ]
            * 5
        ),
    ),
]


def build_synthetic_problem(
    N: int,
    K: int,
    seed: int,
    sectors: list[str],
) -> PortfolioProblem:
    """Build a synthetic portfolio problem with a reproducible random seed."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(N)]
    sector_map = {t: s for t, s in zip(tickers, sectors[:N])}

    # Expected returns ~ U(0.06, 0.32)
    mu = rng.uniform(0.06, 0.32, N)

    # Covariance matrix: factor model + idiosyncratic noise
    n_factors = min(5, N // 2)
    F = rng.normal(0, 0.15, (N, n_factors))
    sigma = (F @ F.T) / n_factors + np.diag(rng.uniform(0.01, 0.04, N))

    return PortfolioProblem.from_arrays(
        tickers=tickers,
        mu=mu,
        sigma=sigma,
        sector_map=sector_map,
        K=K,
        seed=seed,
    )


def main(output_dir: str = "benchmarks/problems") -> None:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Writing benchmark problems to {out.resolve()}\n")

    for name, N, K, seed, sectors in PROBLEM_SPECS:
        t0 = time.perf_counter()
        problem = build_synthetic_problem(N, K, seed, sectors)
        elapsed = time.perf_counter() - t0

        save_path = out / name
        problem.save(str(save_path))

        try:
            n_feasible = math.comb(N, K)
            feasible_str = f"C({N},{K}) = {n_feasible:,}"
        except Exception:
            feasible_str = "too large to enumerate"

        print(
            f"  {name:20s}  N={N:2d}  K={K:d}  "
            f"id={problem.problem_id}  feasible={feasible_str}  "
            f"({elapsed:.2f}s)"
        )

    print(f"\nDone. Problems saved to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark portfolio problems")
    parser.add_argument(
        "--output-dir",
        default="benchmarks/problems",
        help="Directory to save generated problems",
    )
    args = parser.parse_args()
    main(args.output_dir)
