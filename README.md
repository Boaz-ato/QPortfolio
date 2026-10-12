# qportfolio

A research-grade Python library for **classical and quantum multi-objective portfolio optimization**. Implements four classical solvers (Random, WSM, ε-CM, NSGA-II) and a full QAOA pipeline under a shared evaluation framework, enabling mathematically fair classical-vs-quantum comparisons.

## Features

- **Shared objective framework** — all solvers optimize the same three objectives: minimize risk (f1), maximize expected return (f2), maximize diversification (f3)
- **Classical solvers** — Random sampling, Weighted Sum Method (MILP), ε-constraint method (MILP), NSGA-II (pymoo)
- **QAOA solver** — cardinality-preserving XY mixer, Dicke initial state, Qiskit MPS simulator, Julia parameter training
- **Real market data** — built-in yfinance integration with disk caching, annualized returns, and covariance estimation (full or sparse)
- **Reproducible benchmarks** — problem save/load with schema versioning and integrity checks
- **Fair comparison engine** — shared Pareto archive, normalized hypervolume, HV-vs-time curves, comparison tables

## Installation

```bash
# Regular installation
pip install git+https://github.com/Boaz-ato/QPortfolio.git

# Editable installation for developers
git clone https://github.com/Boaz-ato/QPortfolio.git
cd QPortfolio
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, NumPy >= 1.26, pandas >= 2.1, SciPy >= 1.12.

**Julia training** (optional): Install [Julia >= 1.9](https://julialang.org/downloads/) and ensure `julia` is on your PATH. The `qaoa_train.jl` script auto-installs its Julia dependencies (`JSON`, `Optim`) on first run.

## Quick start

```python
import numpy as np
from qportfolio import PortfolioProblem, RandomSampler, WSMSolver, NSGA2Solver

# Build a synthetic problem
rng = np.random.default_rng(42)
N, K = 8, 3
tickers = [f"T{i:02d}" for i in range(N)]
sector_map = {t: ["Tech", "Health", "Finance", "Energy"][i % 4] for i, t in enumerate(tickers)}
mu = rng.uniform(0.08, 0.30, N)
A = rng.normal(0, 0.12, (N, N))
sigma = (A @ A.T) / N + 0.02 * np.eye(N)

problem = PortfolioProblem.from_arrays(
    tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K
)

# Run solvers
results = [
    RandomSampler(n_samples=2000, seed=0).solve(problem),
    WSMSolver(n_weights=50, seed=0).solve(problem),
    NSGA2Solver(generations=50, pop_size=20, seed=0).solve(problem),
]

# Compare
print(problem.compare(results).to_string())
```

### From real market data

```python
from qportfolio import PortfolioProblem

problem = PortfolioProblem.from_yfinance(
    n_stocks=8, n_select=3,
    start="2022-01-01", end="2023-12-31",
    seed=42,
)
problem.save("my_problem")
problem2 = PortfolioProblem.load("my_problem")
assert problem.problem_id == problem2.problem_id
```

### QAOA

```python
from qportfolio import PortfolioProblem, QAOASolver

problem = PortfolioProblem.from_arrays(...)  # N=4, K=2 recommended for testing

qaoa = QAOASolver(p=1, n_weights=20, shots=256, seed=0, n_training_iters=30)
result = qaoa.solve(problem)
print(f"Non-dominated: {result.n_nondominated}, runtime: {result.runtime:.2f}s")
```

### Julia-accelerated training

```python
from qportfolio import QAOASolver, JuliQAOATrainer

trainer = JuliQAOATrainer(optimizer="BFGS")
qaoa = QAOASolver(p=2, n_weights=30, shots=512, trainer=trainer)
result = qaoa.solve(problem)

training = result.quantum["trainer_result"]
print(f"β*={training['params_betas']}, γ*={training['params_gammas']}")
```

## Mathematical formulation

**Objectives** (all under maximization convention):

| Objective | Formula | Meaning |
|-----------|---------|---------|
| f1(x) | -x^T Σ x | Negative risk (maximize = minimize variance) |
| f2(x) | μ^T x | Expected return |
| f3(x) | x^T D x | Diversification score |

**Constraint:** Σx_i = K, x_i ∈ {0,1}

**Diversification matrix:** D_ij = 1 if stocks i,j are in different sectors, 0 otherwise; diagonal = 0.

**Annualization** (when using real market data): μ_i = 252 · mean(r_i), Σ_ij = 252 · Cov(r_i, r_j).



## Solvers

| Solver | Class | Description |
|--------|-------|-------------|
| Random | `RandomSampler` | Uniform random sampling of feasible portfolios |
| WSM | `WSMSolver` | Weighted sum scalarization via HiGHS MILP |
| ε-CM | `EpsilonCMSolver` | ε-constraint method via HiGHS MILP |
| NSGA-II | `NSGA2Solver` | Genetic algorithm with cardinality-preserving operators |
| QAOA | `QAOASolver` | Quantum approximate optimization (Qiskit MPS) |

### QAOA details

- **Initial state:** Uniform Dicke state `|D_K^N⟩` — Bärtschi–Eidenbenz O(KN) construction
- **Mixer:** XY mixer `H_M = Σ_{(i,j)∈E}(X_i X_j + Y_i Y_j)` — preserves Hamming weight exactly, so all samples are feasible
- **Training:** Default is SciPy COBYLA; delegate to Julia (`JuliQAOATrainer`) for BFGS/NelderMead
- **Sampling sweep:** M weight vectors drawn from Dirichlet(1,1,1); trained (β*, γ*) transferred to all vectors without re-optimization per weight
- **Simulator:** `AerSimulator(method="matrix_product_state")` with configurable bond dimension
- **Proxy training:** For N > 12 qubits, parameters are trained on a 12-qubit proxy problem and transferred to the full circuit

### QAOA parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p` | 1 | Circuit depth (layers) |
| `n_weights` | 50 | Weight vectors in sampling sweep |
| `shots` | 1024 | Measurement shots per weight vector |
| `bond_dimension` | 32 | MPS truncation bond dimension |
| `n_training_iters` | 100 | Optimizer iterations for parameter training |
| `trainer` | None | External trainer; falls back to COBYLA when None |

Total shots fired: `n_weights × shots`. `n_evaluations` in the result counts unique feasible bitstrings (always ≤ total shots).

## Evaluation

All solvers share:

- **`ParetoArchive`** — deduplicates, maintains non-dominated set, emits `SolutionDiscoveryEvent`
- **Hypervolume** — normalized (affine transform fixed per-problem before any solver runs) and raw
- **`problem.compare(results)`** — comparison table: algorithm × {HV, n_nondominated, runtime, n_evaluations}
- **Shared reference point** — computed once per problem; comparisons reject mismatched `problem_id`

### Timing

- All timing uses `time.perf_counter()`
- QAOA reports `T_cold = T_training + T_sampling + T_other` and `T_amortized = T_training/R + T_sampling + T_other`

## Problem persistence

```python
# Save
problem.save("path/to/dir")
# Produces: problem.json, mu.npy, sigma.npy, D.npy, [sigma_sparse.npy]

# Load (verifies problem_id integrity and schema version)
problem = PortfolioProblem.load("path/to/dir")
```

`problem_id` is the first 16 hex chars of SHA-256 of the canonical (sorted-ticker) serialization of (tickers, K, mu, Σ, D), rounded to 8 decimal places. It is invariant to ticker insertion order and changes on any numerical perturbation ≥ 1e-8.

## Benchmark datasets

Generate reproducible benchmark instances at three scales:

```bash
python benchmarks/generate_problems.py [--output-dir benchmarks/problems]
```

| Instance | N | K | Feasible portfolios |
|----------|---|---|---------------------|
| N8K3_seed0 | 8 | 3 | C(8,3) = 56 |
| N20K5_seed0 | 20 | 5 | C(20,5) = 15,504 |
| N40K8_seed0 | 40 | 8 | C(40,8) ≈ 76M |

## Examples

```bash
# Classical solver comparison (N=8, K=3)
python examples/01_classical_comparison.py

# QAOA vs random baseline (N=4, K=2)
pip install qportfolio[quantum]
python examples/02_qaoa_example.py

# Fetch real market data and save a problem
python examples/04_fetch_and_save.py

# Full benchmark harness
python examples/05_benchmark.py
```

## Development

```bash
make fmt        # format with ruff
make lint       # ruff check + lint
make typecheck  # mypy
make test       # unit tests (no network, no quantum)
make test-quantum  # Qiskit statevector tests
make test-cov   # coverage report
make test-all   # all except slow tests
```

## Testing

```bash
# Fast unit + regression tests (no network, no quantum)
pytest -q

# Include quantum tests (requires qiskit)
pytest -q -m "quantum"

# Integration tests (hits network)
pytest -q -m "integration"

# Everything except slow tests
pytest -q -m "not slow"
```

## Research experiments

The library is designed to support these comparisons:

- **A:** QAOA vs uniform random — equal samples, equal time, equal evaluations
- **B:** Full vs sparse covariance Hamiltonian — circuit depth, HV, runtime trade-offs
- **C:** QAOA depth p = 1, 2, 3, … scaling
- **D:** Parameter-transfer regret — transferred vs independently optimized parameters per weight vector
- **E:** Julia vs COBYLA training — convergence quality and speed

