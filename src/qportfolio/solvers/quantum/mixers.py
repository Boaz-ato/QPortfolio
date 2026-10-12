"""XY mixer for cardinality-preserving QAOA.

The XY mixer conserves the Hamming weight of all basis states, which means
starting from |D_K^N⟩ the QAOA circuit never leaves the feasible subspace
(sum(x_i) = K).

Hamiltonian: H_M = Σ_{(i,j) ∈ E} (X_i X_j + Y_i Y_j)

where E is the edge set determined by the mixer topology:
  - ``"ring"``  (default): edges (0,1), (1,2), ..., (n-2,n-1), (n-1,0)
  - ``"chain"`` : edges (0,1), (1,2), ..., (n-2,n-1)
  - ``"complete"`` : all pairs (i,j) with i < j

The mixer circuit e^{-i β H_M} is implemented gate-by-gate: each edge (i,j)
contributes an ``fSim``-like rotation (partial iSWAP) with angle β.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp


def _xy_edges(n: int, topology: str) -> list[tuple[int, int]]:
    """Return the edge list for the given topology."""
    if topology == "ring":
        return [(i, (i + 1) % n) for i in range(n)]
    elif topology == "chain":
        return [(i, i + 1) for i in range(n - 1)]
    elif topology == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(
            f"Unknown XY mixer topology: {topology!r}. Choose from 'ring', 'chain', 'complete'."
        )


def xy_mixer_layer(
    n: int,
    beta: Parameter | float,
    topology: str = "ring",
) -> QuantumCircuit:
    """Build one QAOA mixer layer e^{-i β H_M}.

    Each edge (i, j) contributes:
        exp(-i β (X_i X_j + Y_i Y_j)) = iSWAP-like rotation with angle 2β.

    The gate sequence for one edge (i, j) is:
        CNOT(i, j)
        RX(2β) on j
        CNOT(i, j)
    Which implements exp(-i β (XX + YY)) in the {|01⟩, |10⟩} subspace.

    Parameters
    ----------
    n : number of qubits
    beta : mixer angle (Parameter for parametrised circuits, float for fixed)
    topology : edge topology

    Returns
    -------
    QuantumCircuit with no classical bits
    """
    qc = QuantumCircuit(n, name=f"XY_mix({topology})")
    edges = _xy_edges(n, topology)

    from qiskit.circuit.library import XXPlusYYGate

    for i, j in edges:
        # exp(-i β (XX + YY)) on qubits (i, j):
        # Qiskit's XXPlusYYGate(θ) implements:
        #   |01⟩ → cos(θ/2)|01⟩ - i sin(θ/2)|10⟩
        #   |10⟩ → -i sin(θ/2)|01⟩ + cos(θ/2)|10⟩
        # We want cos(2β) and -i sin(2β), so θ = 4β.
        # The β=0 rotational parameter sets the frame (no effect on HW preservation).
        theta = 4 * beta if isinstance(beta, (int, float)) else 4 * beta
        qc.append(XXPlusYYGate(theta, 0.0), [j, i])

    return qc


def xy_mixer_hamiltonian(n: int, topology: str = "ring") -> SparsePauliOp:
    """Build H_M = Σ_{(i,j)∈E} (X_i X_j + Y_i Y_j) as a SparsePauliOp.

    This is the *operator* form of the mixer (used for serialization and matrix
    exponentiation), as opposed to ``xy_mixer_layer`` which builds the circuit.
    """

    edges = _xy_edges(n, topology)
    pauli_list = []
    for i, j in edges:
        xx = ["I"] * n
        xx[i] = "X"
        xx[j] = "X"
        pauli_list.append(("".join(reversed(xx)), 1.0))
        yy = ["I"] * n
        yy[i] = "Y"
        yy[j] = "Y"
        pauli_list.append(("".join(reversed(yy)), 1.0))
    return SparsePauliOp.from_list(pauli_list).simplify()


def verify_xy_commutes_with_cardinality(n: int, topology: str = "ring") -> bool:
    """Verify [H_M, N̂] = 0 for small n using the SparsePauliOp commutator.

    N̂ = Σ_i (I - Z_i)/2 = total particle number operator.

    Returns True if the commutator is zero (up to numerical tolerance).
    """

    edges = _xy_edges(n, topology)

    # Build H_M = Σ_{(i,j)} (XX + YY) in SparsePauliOp
    pauli_list = []
    for i, j in edges:
        # XX on qubits i and j
        xx = ["I"] * n
        xx[i] = "X"
        xx[j] = "X"
        pauli_list.append(("".join(reversed(xx)), 1.0))
        # YY on qubits i and j
        yy = ["I"] * n
        yy[i] = "Y"
        yy[j] = "Y"
        pauli_list.append(("".join(reversed(yy)), 1.0))

    H_M = SparsePauliOp.from_list(pauli_list).simplify()

    # Build N̂ = Σ_i (I - Z_i)/2
    n_hat_list = [("I" * n, float(n) / 2)]
    for q in range(n):
        z_term = ["I"] * n
        z_term[q] = "Z"
        n_hat_list.append(("".join(reversed(z_term)), -0.5))
    N_hat = SparsePauliOp.from_list(n_hat_list).simplify()

    # Commutator [H_M, N̂] = H_M @ N̂ - N̂ @ H_M
    comm = (H_M @ N_hat - N_hat @ H_M).simplify()
    return float(np.max(np.abs(comm.coeffs))) < 1e-10
