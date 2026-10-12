"""Dicke state preparation circuit (uniform fixed-cardinality superposition).

Prepares |D_K^N⟩ = (1/√C(N,K)) Σ_{|x|=K} |x⟩  — the uniform superposition
over all N-bit strings with exactly K ones.

Uses the Bärtschi–Eidenbenz SCS construction (O(KN) 2/3-qubit gates,
polynomial in N and K).  Works for any N.

Reference: Bärtschi & Eidenbenz, "Deterministic Preparation of Dicke States",
FCT 2019.  Algorithm: U_{n,k} = ∏_{l=k+1}^{n} FirstBlock ∘ ∏_{l=2}^{k} SecondBlock,
built from SCS_{n,k} gates composed of (i) and (ii)_l primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class DickeStateMetadata:
    """Circuit metadata returned alongside the Dicke state circuit."""

    n_qubits: int
    k: int
    """Number of ones (excitations / portfolio size)."""
    method: str
    gate_count: int
    depth: int
    n_cx_gates: int


def dicke_state_circuit(
    n: int,
    k: int,
) -> tuple[QuantumCircuit, DickeStateMetadata]:
    """Build a Qiskit circuit that prepares |D_K^N⟩ using the SCS algorithm.

    Parameters
    ----------
    n : number of qubits (N stocks in portfolio)
    k : number of ones (K portfolio size)

    Returns
    -------
    (circuit, metadata)
    """
    if not 0 <= k <= n:
        raise ValueError(f"k={k} must satisfy 0 <= k <= n={n}")
    return _scs_dicke_circuit(n, k)


def _gate_i_scs(n: int):
    """SCS primitive gate (i) — 2-qubit: CNOT, controlled-Ry(2·arccos(√(1/n))), CNOT."""
    from qiskit.circuit.library import RYGate

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    theta = 2 * math.acos(math.sqrt(1.0 / n))
    qc.append(RYGate(theta).control(ctrl_state="1"), [1, 0])
    qc.cx(0, 1)
    return qc.to_gate(label=f"gi({n})")


def _gate_ii_scs(layer_idx: int, n: int):
    """SCS primitive gate (ii)_l — 3-qubit: CNOT, ccRy(2·arccos(√(l/n))), CNOT."""
    from qiskit.circuit.library import RYGate

    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    theta = 2 * math.acos(math.sqrt(layer_idx / n))
    qc.append(RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11"), [2, 1, 0])
    qc.cx(0, 2)
    return qc.to_gate(label=f"gii({layer_idx},{n})")


def _scs_gate(n: int, k: int):
    """SCS_{n,k}: one gate_i(n) on [k-1,k] then k-1 gate_ii_l gates (k+1 qubits)."""
    qc = QuantumCircuit(k + 1)
    qc.append(_gate_i_scs(n), [k - 1, k])
    for layer_idx in range(2, k + 1):
        qc.append(_gate_ii_scs(layer_idx, n), [k - layer_idx, k - layer_idx + 1, k])
    return qc.to_gate(label=f"SCS({n},{k})")


def _scs_first_block(n: int, k: int, layer_idx: int):
    """I^{l-k-1} ⊗ SCS_{l,k} ⊗ I^{n-l} acting on n qubits."""
    qc = QuantumCircuit(n)
    n_first = layer_idx - k - 1
    n_last = n - layer_idx
    idxs = list(range(n))
    if n_first > 0:
        idxs = idxs[n_first:]
    if n_last > 0:
        idxs = idxs[:-n_last]
    qc.append(_scs_gate(layer_idx, k), idxs)
    return qc.to_gate(label=f"FB({n},{k},{layer_idx})")


def _scs_second_block(n: int, k: int, layer_idx: int):
    """SCS_{l,l-1} ⊗ I^{n-l} acting on n qubits."""
    qc = QuantumCircuit(n)
    n_last = n - layer_idx
    idxs = list(range(n))
    if n_last > 0:
        idxs = idxs[:-n_last]
    qc.append(_scs_gate(layer_idx, layer_idx - 1), idxs)
    return qc.to_gate(label=f"SB({n},{k},{layer_idx})")


def _scs_dicke_circuit(n: int, k: int) -> tuple[QuantumCircuit, DickeStateMetadata]:
    """Bärtschi–Eidenbenz SCS circuit — O(KN) gates, works for any N.

    Constructs U_{n,k} from Lemma 2:
      U_{n,k} = [∏_{l=k+1}^{n} FirstBlock(n,k,l)] ∘ [∏_{l=2}^{k} SecondBlock(n,k,l)]
    applied to the initial state |0…0 1…1⟩ (last k qubits set to 1).
    """
    qc = QuantumCircuit(n, name=f"Dicke_SCS({n},{k})")

    if k == 0:
        pass  # |0...0⟩ — nothing to do
    elif k == n:
        qc.x(range(n))  # |1...1⟩
    else:
        # Initial state: set last k qubits to |1⟩
        qc.x(range(n - k, n))

        # First product in Lemma 2: l from n down to k+1
        for layer_idx in range(k + 1, n + 1)[::-1]:
            qc.append(_scs_first_block(n, k, layer_idx), range(n))

        # Second product in Lemma 2: l from k down to 2
        for layer_idx in range(2, k + 1)[::-1]:
            qc.append(_scs_second_block(n, k, layer_idx), range(n))

    qc_decomposed = qc.decompose(reps=4)
    ops = qc_decomposed.count_ops()
    meta = DickeStateMetadata(
        n_qubits=n,
        k=k,
        method="scs",
        gate_count=sum(ops.values()),
        depth=qc_decomposed.depth(),
        n_cx_gates=ops.get("cx", 0),
    )
    return qc, meta


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dicke_circuit(n: int, k: int) -> dict:
    """Validate the Dicke circuit with statevector simulation.

    Checks:
    1. Every non-negligible amplitude corresponds to a bitstring with exactly k ones.
    2. All such amplitudes have equal magnitude (uniform superposition).
    3. The total number of non-zero states equals C(n, k).

    Returns a summary dict.
    """
    circ, meta = dicke_state_circuit(n, k)
    sv = Statevector(circ)
    probs = sv.probabilities_dict(decimals=10)

    expected_count = math.comb(n, k)
    expected_prob = 1.0 / expected_count if expected_count > 0 else 0.0

    threshold = 1e-8
    active = {bs: p for bs, p in probs.items() if p > threshold}

    wrong_hw = [bs for bs in active if bs.count("1") != k]
    prob_errors = [abs(p - expected_prob) for p in active.values()]

    return {
        "n": n,
        "k": k,
        "method": "scs",
        "expected_count": expected_count,
        "actual_count": len(active),
        "count_correct": len(active) == expected_count,
        "hamming_correct": len(wrong_hw) == 0,
        "max_prob_error": max(prob_errors) if prob_errors else 0.0,
        "uniform_correct": all(e < 1e-6 for e in prob_errors),
        "metadata": meta,
    }
