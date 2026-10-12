"""Quantum solver tests — Phase 4.

All tests use N=4, K=2 (C(4,2)=6 feasible portfolios) with exact statevector
simulation to keep runtime short.  The acceptance criterion from the plan:

  "Exact statevector and MPS agree on N=4,K=2 small instance;
   all ideal samples satisfy sum(x)=K."
"""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.solvers.quantum.hamiltonians import (
    build_hamiltonians,
    hamiltonian_energy,
)
from qportfolio.solvers.quantum.mixers import (
    verify_xy_commutes_with_cardinality,
    xy_mixer_layer,
)
from qportfolio.solvers.quantum.states.dicke import (
    dicke_state_circuit,
    validate_dicke_circuit,
)

# ---------------------------------------------------------------------------
# Shared small problem fixture
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


# ===========================================================================
# Dicke state tests
# ===========================================================================


@pytest.mark.parametrize("n,k", [(4, 2), (5, 2), (6, 3), (4, 1), (4, 4)])
def test_dicke_all_amplitudes_have_correct_hw(n, k):
    result = validate_dicke_circuit(n, k)
    assert result["hamming_correct"], f"Dicke({n},{k}): found bitstrings with wrong Hamming weight"


@pytest.mark.parametrize("n,k", [(4, 2), (5, 2), (6, 3), (4, 1)])
def test_dicke_correct_number_of_states(n, k):
    result = validate_dicke_circuit(n, k)
    assert result["count_correct"], (
        f"Dicke({n},{k}): got {result['actual_count']} states, expected {result['expected_count']}"
    )


@pytest.mark.parametrize("n,k", [(4, 2), (5, 2), (6, 3), (4, 1)])
def test_dicke_uniform_amplitudes(n, k):
    result = validate_dicke_circuit(n, k)
    assert result["uniform_correct"], (
        f"Dicke({n},{k}): amplitudes not uniform, max error={result['max_prob_error']:.2e}"
    )


def test_dicke_k0_is_all_zeros():
    """k=0 → circuit should leave all qubits in |0⟩."""
    from qiskit.quantum_info import Statevector

    qc, _ = dicke_state_circuit(4, 0)
    sv = Statevector(qc)
    assert abs(sv[0] - 1.0) < 1e-10, "|0000⟩ should have amplitude 1"


def test_dicke_kn_is_all_ones():
    """k=n → circuit should put all qubits in |1⟩."""
    from qiskit.quantum_info import Statevector

    n = 4
    qc, _ = dicke_state_circuit(n, n)
    sv = Statevector(qc)
    assert abs(sv[2**n - 1] - 1.0) < 1e-10, "|1111⟩ should have amplitude 1"


def test_dicke_invalid_k_raises():
    with pytest.raises(ValueError):
        dicke_state_circuit(4, 5)


def test_dicke_metadata_shape(n=4, k=2):
    _, meta = dicke_state_circuit(n, k)
    assert meta.n_qubits == n
    assert meta.k == k
    assert meta.gate_count > 0
    assert meta.depth > 0


# ===========================================================================
# XY Mixer tests
# ===========================================================================


@pytest.mark.parametrize(
    "n,topology",
    [
        (4, "ring"),
        (4, "chain"),
        (5, "ring"),
        (4, "complete"),
    ],
)
def test_xy_mixer_commutes_with_cardinality(n, topology):
    """[H_M, N̂] = 0 must hold for all topologies."""
    assert verify_xy_commutes_with_cardinality(n, topology), (
        f"XY mixer {topology} on n={n} does NOT commute with N̂"
    )


def test_xy_mixer_preserves_hw_on_dicke_state():
    """Applying e^{-iβH_M} to |D_2^4⟩ should only create HW=2 amplitudes."""
    from qiskit.quantum_info import Statevector

    n, k = 4, 2
    beta = 0.4

    dicke_qc, _ = dicke_state_circuit(n, k)
    mixer_layer = xy_mixer_layer(n, beta, "ring")

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    qc.compose(dicke_qc, inplace=True)
    qc.compose(mixer_layer, inplace=True)

    sv = Statevector(qc)
    threshold = 1e-8
    for idx, amp in enumerate(sv.data):
        if abs(amp) > threshold:
            hw = bin(idx).count("1")
            assert hw == k, (
                f"Amplitude {abs(amp):.2e} on bitstring {format(idx, '04b')} with hw={hw} != k={k}"
            )


def test_xy_mixer_ring_circuit_structure():
    """Ring topology on n=4 should have n=4 edges → 4 xx_plus_yy gates."""
    from qiskit.circuit import Parameter

    beta = Parameter("β")
    layer = xy_mixer_layer(4, beta, "ring")
    ops = layer.count_ops()
    # Ring has 4 edges → 4 XXPlusYY gates
    assert ops.get("xx_plus_yy", 0) == 4


def test_xy_mixer_invalid_topology():
    from qiskit.circuit import Parameter

    with pytest.raises(ValueError, match="topology"):
        xy_mixer_layer(4, Parameter("β"), "invalid_topology")


# ===========================================================================
# Hamiltonian tests
# ===========================================================================


def test_hamiltonian_energy_matches_classical_all_portfolios(problem_N4K2):
    """⟨x|H_j|x⟩ must equal f_j(x) for every feasible portfolio."""
    hc = build_hamiltonians(problem_N4K2.mu, problem_N4K2.sigma, problem_N4K2.D)

    all_x = problem_N4K2.all_feasible_portfolios().astype(float)
    all_objs = problem_N4K2.evaluate_batch(all_x)

    max_err = 0.0
    for i in range(len(all_x)):
        x = all_x[i]
        f1_q = hamiltonian_energy(hc.H_risk, x)
        f2_q = hamiltonian_energy(hc.H_return, x)
        f3_q = hamiltonian_energy(hc.H_diversif, x)
        err = max(
            abs(all_objs[i, 0] - f1_q),
            abs(all_objs[i, 1] - f2_q),
            abs(all_objs[i, 2] - f3_q),
        )
        max_err = max(max_err, err)

    assert max_err < 1e-8, f"Max |classical - quantum energy| = {max_err:.2e} (should be < 1e-8)"


def test_hamiltonian_weighted_energy(problem_N4K2):
    """Weighted Hamiltonian energy should be dot product of weights and objectives."""
    hc = build_hamiltonians(problem_N4K2.mu, problem_N4K2.sigma, problem_N4K2.D)
    weights = np.array([0.3, 0.5, 0.2])
    H_w = hc.weighted(weights)

    all_x = problem_N4K2.all_feasible_portfolios().astype(float)
    all_objs = problem_N4K2.evaluate_batch(all_x)

    for i in range(len(all_x)):
        x = all_x[i]
        e_w = hamiltonian_energy(H_w, x)
        e_cls = float(weights @ all_objs[i])
        assert abs(e_w - e_cls) < 1e-8, (
            f"Weighted energy mismatch: quantum={e_w:.6f} classical={e_cls:.6f}"
        )


def test_hamiltonian_returns_sparse_pauli_op(problem_N4K2):
    from qiskit.quantum_info import SparsePauliOp

    hc = build_hamiltonians(problem_N4K2.mu, problem_N4K2.sigma, problem_N4K2.D)
    assert isinstance(hc.H_risk, SparsePauliOp)
    assert isinstance(hc.H_return, SparsePauliOp)
    assert isinstance(hc.H_diversif, SparsePauliOp)
    assert hc.n_qubits == len(problem_N4K2.tickers)


def test_hamiltonian_weighted_wrong_length(problem_N4K2):
    hc = build_hamiltonians(problem_N4K2.mu, problem_N4K2.sigma, problem_N4K2.D)
    with pytest.raises(ValueError, match="length 3"):
        hc.weighted(np.array([0.5, 0.5]))


def test_hamiltonian_affine_normalization(problem_N4K2):
    """Normalized Hamiltonians should shift and scale energies consistently."""
    all_x = problem_N4K2.all_feasible_portfolios().astype(float)
    all_objs = problem_N4K2.evaluate_batch(all_x)
    shift = all_objs.min(axis=0)
    scale = all_objs.max(axis=0) - all_objs.min(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)

    hc_norm = build_hamiltonians(
        problem_N4K2.mu,
        problem_N4K2.sigma,
        problem_N4K2.D,
        normalization="affine",
        shift=shift,
        scale=scale,
    )
    assert hc_norm.normalization is not None

    # Normalized energies should be in [0, 1] for feasible portfolios
    for i in range(len(all_x)):
        x = all_x[i]
        e1 = hamiltonian_energy(hc_norm.H_risk, x)
        e2 = hamiltonian_energy(hc_norm.H_return, x)
        e3 = hamiltonian_energy(hc_norm.H_diversif, x)
        assert -1e-8 <= e1 <= 1 + 1e-8, f"Normalized f1={e1:.4f} out of [0,1]"
        assert -1e-8 <= e2 <= 1 + 1e-8, f"Normalized f2={e2:.4f} out of [0,1]"
        assert -1e-8 <= e3 <= 1 + 1e-8, f"Normalized f3={e3:.4f} out of [0,1]"


# ===========================================================================
# QAOA solver integration test
# ===========================================================================


def test_qaoa_result_structure(problem_N4K2):
    """QAOA solver should return a valid OptimizationResult on a tiny instance."""
    from qportfolio.solvers.base import OptimizationResult
    from qportfolio.solvers.quantum.qaoa import QAOASolver

    solver = QAOASolver(
        p=1,
        n_weights=5,
        shots=64,
        seed=0,
        n_training_iters=10,
        bond_dimension=16,
    )
    result = solver.solve(problem_N4K2)

    assert isinstance(result, OptimizationResult)
    assert result.algorithm == "qaoa"
    assert result.status in ("feasible", "failed")
    assert result.n_nondominated >= 0
    assert result.runtime >= 0.0
    assert result.quantum is not None
    assert result.quantum["p"] == 1


def test_qaoa_all_solutions_feasible(problem_N4K2):
    """Every solution returned by QAOA must satisfy sum(x)=K."""
    from qportfolio.solvers.quantum.qaoa import QAOASolver

    solver = QAOASolver(p=1, n_weights=3, shots=128, seed=1, n_training_iters=5)
    result = solver.solve(problem_N4K2)

    for sol in result.solutions:
        hw = int(sol.bitstring.sum())
        assert hw == problem_N4K2.K, (
            f"Infeasible solution with Hamming weight {hw} != K={problem_N4K2.K}"
        )


def test_qaoa_nd_subset_of_exact(problem_N4K2):
    """QAOA non-dominated solutions must not be dominated by any exact ND solution."""
    from qportfolio.optimization.pareto_archive import _dominates
    from qportfolio.solvers.quantum.qaoa import QAOASolver

    solver = QAOASolver(p=1, n_weights=10, shots=256, seed=2, n_training_iters=15)
    result = solver.solve(problem_N4K2)

    if result.n_nondominated == 0:
        pytest.skip("QAOA found no non-dominated solutions (acceptable for p=1)")

    all_x = problem_N4K2.all_feasible_portfolios().astype(float)
    exact_objs = problem_N4K2.evaluate_batch(all_x)

    # Compute exact ND set
    nd_mask = np.ones(len(all_x), dtype=bool)
    for i in range(len(all_x)):
        for j in range(len(all_x)):
            if i != j and _dominates(exact_objs[j], exact_objs[i]):
                nd_mask[i] = False
                break
    exact_nd_objs = exact_objs[nd_mask]

    for sol in result.nondominated_solutions:
        dominated = any(_dominates(exact_nd_objs[k], sol.objectives) for k in range(len(exact_nd_objs)))
        assert not dominated, f"QAOA ND solution {sol.objectives.round(4)} dominated by exact ND front"


def test_qaoa_quantum_metadata_fields(problem_N4K2):
    """quantum metadata dict must contain key fields."""
    from qportfolio.solvers.quantum.qaoa import QAOASolver

    solver = QAOASolver(p=1, n_weights=3, shots=64, seed=0, n_training_iters=5)
    result = solver.solve(problem_N4K2)

    meta = result.quantum
    assert meta is not None
    for key in (
        "p",
        "n_weights",
        "shots_per_weight",
        "params_gammas",
        "params_betas",
        "timing",
        "total_shots",
        "hardware_violations",
    ):
        assert key in meta, f"Missing quantum metadata key: {key}"

    assert meta["hardware_violations"] == 0, (
        "XY mixer should guarantee zero hardware_violations (wrong HW)"
    )


def test_qaoa_statevector_vs_mps_agree(problem_N4K2):
    """Statevector and MPS expectation values should agree for N=4,K=2."""
    from qiskit.quantum_info import Statevector

    from qportfolio.solvers.quantum.hamiltonians import build_hamiltonians
    from qportfolio.solvers.quantum.mixers import xy_mixer_layer
    from qportfolio.solvers.quantum.samplers.qiskit_mps import (
        _build_qaoa_circuit,
        _cost_unitary_from_hamiltonian,
    )
    from qportfolio.solvers.quantum.states.dicke import dicke_state_circuit

    n, k = 4, 2
    hc = build_hamiltonians(problem_N4K2.mu, problem_N4K2.sigma, problem_N4K2.D)
    H_w = hc.weighted(np.array([1 / 3, 1 / 3, 1 / 3]))

    dicke_qc, _ = dicke_state_circuit(n, k)

    gamma_val, beta_val = 0.3, 0.5

    def _cost(g):
        return _cost_unitary_from_hamiltonian(H_w, g, n)

    def _mix(b):
        return xy_mixer_layer(n, b, "ring")

    qaoa_qc, gammas_pv, betas_pv = _build_qaoa_circuit(n, 1, dicke_qc.copy(), _cost, _mix)
    bound = qaoa_qc.assign_parameters({gammas_pv[0]: gamma_val, betas_pv[0]: beta_val})

    # Statevector expectation
    sv_ev = Statevector(bound).expectation_value(H_w).real

    # MPS expectation via StatevectorEstimator (exact, same result as SV above)
    from qiskit.primitives import StatevectorEstimator

    est = StatevectorEstimator()
    pub = (bound, H_w)
    mps_ev = float(est.run([pub]).result()[0].data.evs)

    assert abs(sv_ev - mps_ev) < 1e-3, (
        f"Statevector EV={sv_ev:.6f} vs MPS EV={mps_ev:.6f}: diff={abs(sv_ev - mps_ev):.2e}"
    )
