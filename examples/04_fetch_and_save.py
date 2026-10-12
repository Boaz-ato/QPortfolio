"""Example 4: Fetch market data from Yahoo Finance and save a PortfolioProblem.

Downloads historical price data for a universe of stocks, builds the three
objective matrices (mu, sigma, D), and saves the problem to disk for reuse
in benchmarks or solver runs.

Three usage modes are shown:
  - sector_balanced: auto-select N stocks spread evenly across sectors
  - random:          randomly sample N stocks from the universe registry
  - custom:          supply your own ticker list

Usage:
    python examples/04_fetch_and_save.py
    python examples/04_fetch_and_save.py --mode custom --tickers AAPL MSFT GOOGL AMZN META NVDA --K 3
    python examples/04_fetch_and_save.py --output-dir my_problems --n-stocks 20 --K 5
"""

from __future__ import annotations

import argparse
import pathlib

from qportfolio import PortfolioProblem


def fetch_and_save(
    output_dir: str,
    n_stocks: int,
    K: int,
    start: str,
    end: str,
    seed: int,
    mode: str,
    tickers: list[str] | None,
    covariance: str,
) -> PortfolioProblem:
    print(f"Fetching data from Yahoo Finance ({start} → {end})...")
    print(f"  Universe mode : {mode}")
    if tickers:
        print(f"  Tickers       : {tickers}")
    else:
        print(f"  N stocks      : {n_stocks}")
    print(f"  K (select)    : {K}")
    print(f"  Covariance    : {covariance}")
    print()

    problem = PortfolioProblem.from_yfinance(
        n_stocks=n_stocks,
        n_select=K,
        start=start,
        end=end,
        seed=seed,
        universe_mode=mode,
        custom_tickers=tickers,
        covariance=covariance,
    )

    print("Problem built:")
    print(f"  ID      : {problem.problem_id}")
    print(f"  Tickers : {problem.tickers}")
    print(f"  N={problem.N}, K={problem.K}")
    print(f"  Date range: {problem.actual_start} → {problem.actual_end}")
    print()

    save_path = pathlib.Path(output_dir) / problem.problem_id
    problem.save(str(save_path))
    print(f"Saved to: {save_path.resolve()}")
    print("  Files: problem.json, mu.npy, sigma.npy, D.npy")

    return problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch market data and save a PortfolioProblem")
    parser.add_argument(
        "--output-dir", default="examples/saved_problems", help="Directory to save the problem"
    )
    parser.add_argument(
        "--n-stocks",
        type=int,
        default=8,
        help="Number of stocks in the universe (ignored for custom mode)",
    )
    parser.add_argument("--K", type=int, default=3, help="Number of stocks to select per portfolio")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for universe selection")
    parser.add_argument(
        "--mode",
        choices=["sector_balanced", "random", "custom"],
        default="sector_balanced",
        help="Universe selection mode",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols (required for --mode custom)",
    )
    parser.add_argument(
        "--covariance",
        choices=["full", "sparse"],
        default="full",
        help="Covariance estimator type",
    )
    args = parser.parse_args()

    if args.mode == "custom" and not args.tickers:
        parser.error("--tickers is required when --mode=custom")

    fetch_and_save(
        output_dir=args.output_dir,
        n_stocks=args.n_stocks,
        K=args.K,
        start=args.start,
        end=args.end,
        seed=args.seed,
        mode=args.mode,
        tickers=args.tickers,
        covariance=args.covariance,
    )


if __name__ == "__main__":
    main()
