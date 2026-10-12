"""Phase 5 tests — Julia training adapter.

Mandatory compatibility test (from the design spec):
  "Julia-trained parameters reproduce same circuit expectation value
   in Qiskit to within 1e-4."

All tests that require Julia are decorated with ``@pytest.mark.julia`` and
are skipped automatically when Julia is not installed.  This lets CI run the
full test suite without Julia while still exercising the Python-side API.

Python-only tests
-----------------
- TrainingResult dataclass and to_dict()
- JuliQAOATrainer.hamiltonian_to_pauli_terms()
- JuliQAOATrainer.is_available() returns a bool
- xy_mixer_hamiltonian() builds a correct SparsePauliOp
- trainers/__init__.py re-exports

Julia-required tests (skipped if Julia absent)
----------------------------------------------
- Julia script produces parseable output for N=4, K=2
- Trained parameters give correct expectation value in Qiskit (≤ 1e-4 error)
- hw_violations == 0 when QAOASolver uses JuliQAOATrainer
- JuliQAOATrainer raises RuntimeError with a clear message on bad inputs
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.solvers.quantum.hamiltonians import build_hamiltonians
from qportfolio.solvers.quantum.mixers import xy_mixer_hamiltonian
from qportfolio.solvers.quantum.trainers.base import BaseTrainer, TrainingResult
from qportfolio.solvers.quantum.trainers.juliqaoa import (
    _JULIA_SCRIPT,
    JuliQAOATrainer,
)

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def problem_N4K2():
    rng = np.random.default_rng(42)
    N, K = 4, 2
    tickers = [f"S{i}" for i in range(N)]
    sectors = ["Tech", "Health", "Tech", "Health"]
    sector_map = dict(zip(tickers, sectors))
    mu = rng.uniform(0.08, 0.28, N)
    A = rng.normal(0, 0.05, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)
    return PortfolioProblem.from_arrays(tickers=tickers, mu=mu, sigma=sigma, sector_map=sector_map, K=K)


@pytest.fixture(scope="module")
def hc_N4K2(problem_N4K2):
    return build_hamiltonians(problem_N4K2.mu, problem_N4K2.sigma, problem_N4K2.D)


julia_available = pytest.mark.skipif(
    not JuliQAOATrainer.is_available(),
    reason="Julia not installed or qaoa_train.jl not found",
)


# ---------------------------------------------------------------------------
# Python-only: TrainingResult
# ---------------------------------------------------------------------------


def test_training_result_to_dict():
    tr = TrainingResult(
        params_gammas=np.array([0.5]),
        params_betas=np.array([-0.3]),
        history=[{"iter": 1, "energy": 0.1}],
        n_iterations=10,
        final_energy=0.42,
        training_time=1.23,
        converged=True,
        backend="test_backend",
    )
    d = tr.to_dict()
    assert d["backend"] == "test_backend"
    assert d["params_gammas"] == [0.5]
    assert d["params_betas"] == [-0.3]
    assert d["converged"] is True
    assert d["n_iterations"] == 10
    assert abs(d["final_energy"] - 0.42) < 1e-12
    assert abs(d["training_time"] - 1.23) < 1e-12


def test_training_result_defaults():
    tr = TrainingResult(
        params_gammas=np.zeros(1),
        params_betas=np.zeros(1),
    )
    assert tr.converged is False
    assert tr.history == []
    assert tr.backend == "unknown"


# ---------------------------------------------------------------------------
# Python-only: BaseTrainer ABC
# ---------------------------------------------------------------------------


def test_base_trainer_is_abstract():
    with pytest.raises(TypeError):
        BaseTrainer()  # type: ignore[abstract]


def test_base_trainer_default_available():
    class _Concrete(BaseTrainer):
        @property
        def name(self):
            return "test"

        def train(self, *args, **kwargs):
            return TrainingResult(np.zeros(1), np.zeros(1))

    assert _Concrete.is_available() is True


# ---------------------------------------------------------------------------
# Python-only: JuliQAOATrainer API
# ---------------------------------------------------------------------------


def test_juliqaoa_name():
    t = JuliQAOATrainer(optimizer="BFGS")
    assert "bfgs" in t.name.lower()

    t2 = JuliQAOATrainer(optimizer="NelderMead")
    assert "neldermead" in t2.name.lower()


def test_juliqaoa_is_available_returns_bool():
    result = JuliQAOATrainer.is_available()
    assert isinstance(result, bool)


def test_juliqaoa_invalid_optimizer():
    with pytest.raises(ValueError, match="optimizer"):
        JuliQAOATrainer(optimizer="gradient_descent")


def test_juliqaoa_julia_script_exists():
    """The bundled Julia script must exist in the installed package."""
    assert _JULIA_SCRIPT.exists(), f"Julia script not found: {_JULIA_SCRIPT}"


def test_juliqaoa_hamiltonian_to_pauli_terms(hc_N4K2):
    terms = JuliQAOATrainer.hamiltonian_to_pauli_terms(hc_N4K2.H_risk)
    assert len(terms) > 0
    for label, coeff in terms:
        assert isinstance(label, str)
        assert len(label) == hc_N4K2.n_qubits
        assert all(c in "IXYZ" for c in label)
        assert isinstance(coeff, complex)


def test_juliqaoa_hamiltonian_to_pauli_terms_weighted(hc_N4K2):
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    H_w = hc_N4K2.weighted(weights)
    terms = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_w)
    assert len(terms) > 0
    # Coefficients should be real (imaginary part negligible for our Hamiltonians)
    for _, coeff in terms:
        assert abs(coeff.imag) < 1e-10, f"Non-real coefficient: {coeff}"


# ---------------------------------------------------------------------------
# Python-only: xy_mixer_hamiltonian
# ---------------------------------------------------------------------------


def test_xy_mixer_hamiltonian_type():
    from qiskit.quantum_info import SparsePauliOp

    H_M = xy_mixer_hamiltonian(4, "ring")
    assert isinstance(H_M, SparsePauliOp)
    assert H_M.num_qubits == 4


def test_xy_mixer_hamiltonian_terms():
    """Ring on n=4 has 4 edges → 4 XX + 4 YY = 8 Pauli terms (before simplification)."""
    H_M = xy_mixer_hamiltonian(4, "ring")
    terms = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_M)
    assert len(terms) > 0
    # All coefficients should be real and positive (XX and YY have coeff +1)
    for _, coeff in terms:
        assert abs(coeff.imag) < 1e-10


def test_xy_mixer_hamiltonian_commutes(hc_N4K2):
    """[H_M, N̂] = 0 should hold for the SparsePauliOp built by xy_mixer_hamiltonian."""
    from qiskit.quantum_info import SparsePauliOp

    n = hc_N4K2.n_qubits
    H_M = xy_mixer_hamiltonian(n, "ring")

    n_hat_list = [("I" * n, float(n) / 2)]
    for q in range(n):
        z_term = ["I"] * n
        z_term[q] = "Z"
        n_hat_list.append(("".join(reversed(z_term)), -0.5))
    N_hat = SparsePauliOp.from_list(n_hat_list).simplify()

    comm = (H_M @ N_hat - N_hat @ H_M).simplify()
    assert float(np.max(np.abs(comm.coeffs))) < 1e-10


# ---------------------------------------------------------------------------
# Python-only: trainers __init__ re-exports
# ---------------------------------------------------------------------------


def test_trainers_init_exports():
    from qportfolio.solvers.quantum.trainers import (
        BaseTrainer,
        JuliQAOATrainer,
        TrainingResult,
    )

    assert BaseTrainer is not None
    assert JuliQAOATrainer is not None
    assert TrainingResult is not None


# ---------------------------------------------------------------------------
# Julia tests — skipped if Julia not available
# ---------------------------------------------------------------------------


@julia_available
def test_julia_script_runs_and_outputs_json(hc_N4K2, problem_N4K2):
    """Julia script should run without error and produce valid JSON output."""
    import shutil
    import subprocess

    julia = shutil.which("julia")
    n = hc_N4K2.n_qubits
    p = 1

    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    H_cost_op = hc_N4K2.weighted(weights)
    H_mixer_op = xy_mixer_hamiltonian(n, "ring")

    cost_terms = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_cost_op)
    mixer_terms = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_mixer_op)

    payload = {
        "pauli_terms_cost": [
            {"label": lbl, "coeff_re": float(c.real), "coeff_im": float(c.imag)} for lbl, c in cost_terms
        ],
        "pauli_terms_mixer": [
            {"label": lbl, "coeff_re": float(c.real), "coeff_im": float(c.imag)}
            for lbl, c in mixer_terms
        ],
        "n": n,
        "k": problem_N4K2.K,
        "p": p,
        "n_iters": 20,
        "x0": [0.3, 0.5],
        "seed": 0,
        "optimizer": "NelderMead",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.json")
        out_path = os.path.join(tmpdir, "output.json")
        with open(in_path, "w") as f:
            json.dump(payload, f)

        proc = subprocess.run(
            [julia, str(_JULIA_SCRIPT), in_path, out_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"Julia script failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
        assert os.path.exists(out_path), "Julia did not produce output file"

        with open(out_path) as f:
            result = json.load(f)

    assert "params_gammas" in result
    assert "params_betas" in result
    assert "final_energy" in result
    assert len(result["params_gammas"]) == p
    assert len(result["params_betas"]) == p
    assert isinstance(result["converged"], bool)


@julia_available
def test_julia_params_reproduce_qiskit_expectation(hc_N4K2, problem_N4K2):
    """Acceptance criterion: Julia-trained params reproduce Qiskit EV to within 1e-4."""
    from qiskit.primitives import StatevectorEstimator

    from qportfolio.solvers.quantum.samplers.qiskit_mps import (
        _build_qaoa_circuit,
        _cost_unitary_from_hamiltonian,
    )
    from qportfolio.solvers.quantum.states.dicke import dicke_state_circuit

    n = hc_N4K2.n_qubits
    k = problem_N4K2.K
    p = 1
    weights = np.array([1 / 3, 1 / 3, 1 / 3])

    trainer = JuliQAOATrainer(n_iters=50, optimizer="NelderMead")

    H_cost_op = hc_N4K2.weighted(weights)
    H_mixer_op = xy_mixer_hamiltonian(n, "ring")
    cost_terms = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_cost_op)
    mixer_terms = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_mixer_op)

    rng = np.random.default_rng(0)
    x0 = rng.uniform(-np.pi, np.pi, 2 * p)

    train_result = trainer.train(
        H_cost_pauli=cost_terms,
        H_mixer_pauli=mixer_terms,
        n=n,
        k=k,
        p=p,
        n_iters=50,
        x0=x0,
        seed=0,
    )

    assert train_result.params_gammas.shape == (p,)
    assert train_result.params_betas.shape == (p,)

    # Reproduce the expectation value in Qiskit using the Julia-trained params
    dicke_qc, _ = dicke_state_circuit(n, k)
    H_w = H_cost_op

    from qportfolio.solvers.quantum.mixers import xy_mixer_layer

    def _cost(g):
        return _cost_unitary_from_hamiltonian(H_w, g, n)

    def _mix(b):
        return xy_mixer_layer(n, b, "ring")

    qaoa_qc, gammas_pv, betas_pv = _build_qaoa_circuit(n, p, dicke_qc.copy(), _cost, _mix)
    param_dict = {
        gammas_pv[0]: float(train_result.params_gammas[0]),
        betas_pv[0]: float(train_result.params_betas[0]),
    }
    bound = qaoa_qc.assign_parameters(param_dict)

    estimator = StatevectorEstimator()
    pub = (bound, H_w)
    qiskit_ev = float(estimator.run([pub]).result()[0].data.evs)

    # Julia's final_energy should agree with Qiskit's EV to within 1e-4
    diff = abs(train_result.final_energy - qiskit_ev)
    assert diff < 1e-4, (
        f"Julia energy={train_result.final_energy:.8f}, "
        f"Qiskit EV={qiskit_ev:.8f}, diff={diff:.2e} (threshold 1e-4)"
    )


@julia_available
def test_julia_trainer_integration_with_qaoa_solver(problem_N4K2):
    """QAOASolver with JuliQAOATrainer should return valid result with 0 hw_violations."""
    from qportfolio.solvers.quantum.qaoa import QAOASolver

    trainer = JuliQAOATrainer(n_iters=20, optimizer="NelderMead")
    solver = QAOASolver(
        p=1,
        n_weights=3,
        shots=64,
        seed=0,
        n_training_iters=20,
        trainer=trainer,
    )
    result = solver.solve(problem_N4K2)

    assert result.algorithm == "qaoa"
    assert result.quantum is not None
    assert result.quantum["trainer"] == trainer.name
    assert result.quantum["hardware_violations"] == 0

    for sol in result.solutions:
        hw = int(sol.bitstring.sum())
        assert hw == problem_N4K2.K


@julia_available
def test_julia_trainer_bad_executable_raises():
    """JuliQAOATrainer should raise RuntimeError if julia binary not found."""
    trainer = JuliQAOATrainer(julia_executable="julia_does_not_exist_xyz")
    with pytest.raises(RuntimeError, match="Julia executable not found"):
        trainer.train(
            H_cost_pauli=[("IIII", 1.0 + 0j)],
            H_mixer_pauli=[("IIII", 1.0 + 0j)],
            n=4,
            k=2,
            p=1,
            n_iters=5,
            x0=np.zeros(2),
            seed=0,
        )
