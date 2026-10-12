# qportfolio

A research-grade Python library for **classical and quantum multi-objective portfolio optimization**. Implements four classical solvers (Random, WSM, ε-CM, NSGA-II) and a full QAOA pipeline under a shared evaluation framework, enabling mathematically fair classical-vs-quantum comparisons.

## Features

- **Shared objective framework** — all solvers optimize the same three objectives: minimize risk (f1), maximize expected return (f2), maximize diversification (f3)
- **Classical solvers** — Random sampling, Weighted Sum Method (MILP), ε-constraint method (MILP), NSGA-II (pymoo)
- **QAOA solver** — cardinality-preserving XY mixer, Dicke initial state, Qiskit MPS simulator, Julia parameter training
- **Warm-start QAOA** — SLSQP classical relaxation → weighted Dicke initial state preparation
- **Reproducible benchmarks** — problem save/load with schema versioning and integrity checks
- **Fair comparison engine** — shared Pareto archive, normalized hypervolume, HV-vs-time curves, comparison tables

## Installation

# Regular installation
pip install git+https://github.com/Boaz-ato/QPortfolio.git

# Editable installation for developers
git clone https://github.com/Boaz-ato/QPortfolio.git
cd QPortfolio
pip install -e ".[dev]"

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

### Warm-start QAOA

```python
from qportfolio import QAOASolver

qaoa_ws = QAOASolver(
    p=1, n_weights=20, shots=256, seed=0, n_training_iters=30,
    warm_start=True, warm_start_n_restarts=5,
)
result = qaoa_ws.solve(problem)

# Inspect relaxation result
ws_meta = result.quantum["warm_start_meta"]
z_star = ws_meta["relaxation"]["z_star"]
print(f"SLSQP z*: {[f'{z:.3f}' for z in z_star]}")
```

## Mathematical formulation

**Objectives** (all under maximization convention):

| Objective | Formula | Meaning |
|-----------|---------|---------|
| f1(x) | -x^T Sigma x | Negative risk (maximize = minimize variance) |
| f2(x) | mu^T x | Expected return |
| f3(x) | x^T D x | Diversification score |

**Constraint:** sum(x_i) = K, x_i in {0,1}

**Diversification matrix:** D_ij = 1 if stocks i,j are in different sectors, 0 otherwise; diagonal = 0.

**Annualization** (when using real market data): mu_i = 252 * mean(r_i), Sigma_ij = 252 * Cov(r_i, r_j).

## Architecture

```
src/qportfolio/
├── data/           # MarketDataProvider (yfinance), CovarianceEstimator, DiskCache
├── problems/       # PortfolioProblem, ObjectiveEvaluator, ConstraintValidator
├── solvers/
│   ├── random.py   # Random sampling baseline
│   ├── wsm.py      # Weighted Sum Method (HiGHS MILP)
│   ├── epsilon.py  # epsilon-constraint method (HiGHS MILP)
│   ├── nsga2.py    # NSGA-II (pymoo)
│   └── quantum/
│       ├── qaoa.py             # QAOASolver (orchestrator)
│       ├── hamiltonians.py     # Pauli Hamiltonian builder
│       ├── mixers.py           # XY mixer circuits + Hamiltonians
│       ├── states/
│       │   ├── dicke.py            # Uniform Dicke state (Bartschi-Eidenbenz)
│       │   └── weighted_dicke.py   # Weighted Dicke state (warm-start)
│       ├── warm_start.py       # SLSQP classical relaxation
│       ├── trainers/
│       │   ├── base.py         # BaseTrainer ABC, TrainingResult
│       │   └── juliqaoa.py     # Julia subprocess bridge
│       └── samplers/
│           └── qiskit_mps.py   # Qiskit MPS simulator
├── optimization/   # ParetoArchive, MILP backend, weight sampling
├── evaluation/     # Hypervolume, benchmarking, timing
└── results/        # OptimizationResult models, serialization
```

## Solvers

| Solver | Class | Description |
|--------|-------|-------------|
| Random | `RandomSampler` | Uniform random sampling of feasible portfolios |
| WSM | `WSMSolver` | Weighted sum scalarization via HiGHS MILP |
| epsilon-CM | `EpsilonCMSolver` | epsilon-constraint method via HiGHS MILP |
| NSGA-II | `NSGA2Solver` | Genetic algorithm with cardinality-preserving operators |
| QAOA | `QAOASolver` | Quantum approximate optimization (Qiskit MPS) |

### QAOA details

- **Initial state:** Uniform Dicke state `|D_N^K>` (or weighted Dicke for warm-start)
- **Mixer:** XY mixer `H_M = sum_{(i,j) in E}(X_i X_j + Y_i Y_j)` — preserves Hamming weight exactly
- **Training:** Default is SciPy COBYLA; optionally delegate to Julia (`JuliQAOATrainer`) for BFGS/NelderMead
- **Sampling sweep:** M weight vectors drawn from Dirichlet(1,1,1); trained beta*, gamma* transferred to all vectors (no re-optimization per weight vector)
- **Simulator:** `AerSimulator(method="matrix_product_state")` with configurable bond dimension

### Julia training (optional)

```python
from qportfolio import QAOASolver, JuliQAOATrainer

trainer = JuliQAOATrainer(optimizer="BFGS")
qaoa = QAOASolver(p=2, n_weights=30, shots=512, trainer=trainer)
result = qaoa.solve(problem)

# Training metadata
training = result.quantum["trainer_result"]
print(f"Julia trained betas={training['params_betas']}, gammas={training['params_gammas']}")
```

## Evaluation

All solvers share:

- **`ParetoArchive`** — deduplicates, maintains non-dominated set, emits `SolutionDiscoveryEvent`
- **Hypervolume** — normalized (affine transform fixed per-problem before any solver runs) and raw
- **`problem.compare(results)`** — comparison table: algorithm x {HV, n_nondominated, runtime, n_evaluations}
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

`problem_id` is the first 16 hex chars of SHA-256 of the canonical (sorted-ticker) serialization of (tickers, K, mu, Sigma, D), rounded to 8 decimal places. It is invariant to ticker insertion order and changes on any numerical perturbation >= 1e-8.

## Benchmark datasets

Generate reproducible benchmark instances at three scales:

```bash
python benchmarks/generate_problems.py [--output-dir benchmarks/problems]
```

| Instance | N | K | Feasible portfolios |
|----------|---|---|---------------------|
| N8K3_seed0 | 8 | 3 | C(8,3) = 56 |
| N20K5_seed0 | 20 | 5 | C(20,5) = 15,504 |
| N40K8_seed0 | 40 | 8 | C(40,8) ~76M |

## Examples

```bash
# Classical solver comparison (N=8, K=3)
python examples/01_classical_comparison.py

# QAOA vs random baseline (N=4, K=2)
pip install qportfolio[quantum]
python examples/02_qaoa_example.py

# Warm-start vs standard QAOA comparison
python examples/03_warm_start.py
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

## CI

GitHub Actions runs on every push and pull request:

| Job | What it checks |
|-----|---------------|
| `lint` | `ruff check` + `mypy` |
| `test` | All non-quantum, non-slow tests on Python 3.10/3.11/3.12 |
| `quantum` | Statevector QAOA tests (Qiskit, no Julia) |
| `packaging` | Wheel builds and imports cleanly |

## Key design invariants

1. **Single evaluator** — all objectives always use the full (not sparse) Sigma. QAOA may use sparse Sigma for its Hamiltonian but solution quality is always measured with the full covariance.
2. **Shared reference point** — computed before any solver runs; `OptimizationResult` stores `problem_id`; direct HV comparisons reject mismatched ids.
3. **Weight vector consistency** — both WSM and QAOA use `SimplexWeightSampler` (Dirichlet(1,1,1)) with the same seed when comparing performance.
4. **QAOA total runtime** — always includes training time; `T_cold` and `T_amortized` both reported.
5. **Cardinality preservation** — NSGA-II operators and XY mixer both guarantee sum(x_i) = K without post-hoc repair.

## Research experiments

The library is designed to support these comparisons from the reference paper:

- **A:** QAOA vs uniform random — equal samples, equal time, equal evaluations
- **B:** Warm-start vs standard QAOA — same instances, weight vectors, shots
- **C:** Full vs sparse covariance Hamiltonian — circuit depth, HV, runtime trade-offs
- **D:** QAOA depth p = 1, 2, 3 scaling
- **E:** Parameter-transfer regret — transferred vs independently optimized parameters per weight vector

## License

MIT
