"""Hamiltonian builder for multi-objective portfolio QAOA.

Pauli mapping: x_i → (I - Z_i)/2

All three objectives use maximization convention (same as the classical solvers):
  H_risk        → encodes f1 = -x^T Σ x
  H_return      → encodes f2 =  μ^T x
  H_diversif    → encodes f3 =  x^T D x

The weighted cost Hamiltonian for a weight vector c = (c1, c2, c3):
  H_C(c) = c1 H_risk + c2 H_return + c3 H_diversif

Computational-basis expectation value of H_C(c) on bitstring x must equal
c · (f1(x), f2(x), f3(x)) — this is verified in the tests.

Normalization:
  Optionally normalizes each objective Hamiltonian using problem-level shift/scale
  so that all objectives contribute comparably to the optimizer.
  Raw objectives are always used for Pareto/HV evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass
class HamiltonianComponents:
    """The three individual objective Hamiltonians and the combined cost."""

    n_qubits: int
    H_risk: SparsePauliOp  # encodes f1 = -x^T Σ x
    H_return: SparsePauliOp  # encodes f2 = μ^T x
    H_diversif: SparsePauliOp  # encodes f3 = x^T D x
    normalization: dict | None = field(default=None)
    """dict with keys 'shift' (3,) and 'scale' (3,) if normalization was applied."""

    def weighted(self, weights: np.ndarray) -> SparsePauliOp:
        """Build H_C(c) = c1*H_risk + c2*H_return + c3*H_diversif.

        Parameters
        ----------
        weights : (3,) array with c1+c2+c3 = 1, all ≥ 0.
        """
        w = np.asarray(weights, dtype=float)
        if len(w) != 3:
            raise ValueError(f"weights must have length 3, got {len(w)}")
        return (w[0] * self.H_risk + w[1] * self.H_return + w[2] * self.H_diversif).simplify()


def _qubit_to_pauli(n: int, q: int, gate: str) -> SparsePauliOp:
    """Single-qubit Pauli on qubit q in an n-qubit system."""
    # Qiskit SparsePauliOp uses little-endian convention (qubit 0 = rightmost).
    label = ["I"] * n
    label[q] = gate  # position q in 0-indexed = rightmost char for q=0
    return SparsePauliOp("".join(reversed(label)))


def _x_i(n: int, q: int) -> SparsePauliOp:
    """x_i = (I - Z_i)/2 as a SparsePauliOp."""
    I_op = SparsePauliOp("I" * n)
    Z_i = _qubit_to_pauli(n, q, "Z")
    return ((I_op - Z_i) * 0.5).simplify()


def build_hamiltonians(
    mu: np.ndarray,
    sigma: np.ndarray,
    D: np.ndarray,
    normalization: str = "none",
    shift: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> HamiltonianComponents:
    """Build the three objective Hamiltonians for a portfolio problem.

    Parameters
    ----------
    mu : (N,) expected returns
    sigma : (N, N) covariance matrix (full, symmetric, PSD)
    D : (N, N) diversification matrix (0/1, zero diagonal)
    normalization : ``"none"`` (default) or ``"affine"``
        ``"affine"`` divides each Hamiltonian by ``scale[j]`` and subtracts
        ``shift[j]/scale[j]`` after Pauli construction.
    shift, scale : required when normalization="affine"; shapes (3,)

    Returns
    -------
    HamiltonianComponents
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    D = np.asarray(D, dtype=float)
    N = len(mu)

    # ------------------------------------------------------------------
    # H_return = Σ_i μ_i x_i
    # ------------------------------------------------------------------
    H_ret = SparsePauliOp("I" * N, coeffs=[0.0])
    for i in range(N):
        H_ret = (H_ret + mu[i] * _x_i(N, i)).simplify()

    # ------------------------------------------------------------------
    # H_risk = -Σ_{i,j} Σ_ij x_i x_j = -x^T Σ x
    # x_i x_j = (I-Z_i)(I-Z_j)/4 for i ≠ j
    # x_i^2   = x_i  (binary)
    # ------------------------------------------------------------------
    H_rsk = SparsePauliOp("I" * N, coeffs=[0.0])
    for i in range(N):
        for j in range(N):
            coeff = -sigma[i, j]
            if i == j:
                H_rsk = (H_rsk + coeff * _x_i(N, i)).simplify()
            else:
                xi = _x_i(N, i)
                xj = _x_i(N, j)
                H_rsk = (H_rsk + coeff * (xi @ xj)).simplify()

    # ------------------------------------------------------------------
    # H_diversif = Σ_{i,j} D_ij x_i x_j = x^T D x
    # ------------------------------------------------------------------
    H_div = SparsePauliOp("I" * N, coeffs=[0.0])
    for i in range(N):
        for j in range(N):
            coeff = D[i, j]
            if abs(coeff) < 1e-14:
                continue
            if i == j:
                H_div = (H_div + coeff * _x_i(N, i)).simplify()
            else:
                xi = _x_i(N, i)
                xj = _x_i(N, j)
                H_div = (H_div + coeff * (xi @ xj)).simplify()

    # ------------------------------------------------------------------
    # Optional affine normalization
    # ------------------------------------------------------------------
    norm_info: dict | None = None
    if normalization == "affine":
        if shift is None or scale is None:
            raise ValueError("shift and scale must be provided for normalization='affine'.")
        shift = np.asarray(shift, dtype=float)
        scale = np.asarray(scale, dtype=float)
        if np.any(scale <= 0):
            raise ValueError("All scale values must be positive.")
        # Normalized: H̃_j = (H_j - shift_j) / scale_j
        H_rsk = ((H_rsk - shift[0] * SparsePauliOp("I" * N)) * (1.0 / scale[0])).simplify()
        H_ret = ((H_ret - shift[1] * SparsePauliOp("I" * N)) * (1.0 / scale[1])).simplify()
        H_div = ((H_div - shift[2] * SparsePauliOp("I" * N)) * (1.0 / scale[2])).simplify()
        norm_info = {"shift": shift, "scale": scale}

    return HamiltonianComponents(
        n_qubits=N,
        H_risk=H_rsk,
        H_return=H_ret,
        H_diversif=H_div,
        normalization=norm_info,
    )


def hamiltonian_energy(
    hamiltonian: SparsePauliOp,
    bitstring: np.ndarray,
) -> float:
    """Evaluate the computational-basis energy ⟨x|H|x⟩ for a given bitstring.

    Used to verify that the Pauli mapping gives energies matching the classical
    objective evaluator.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
    bitstring : (N,) binary array (0/1)

    Returns
    -------
    float — expectation value ⟨x|H|x⟩
    """

    n = hamiltonian.num_qubits
    x = np.asarray(bitstring, dtype=int)
    if len(x) != n:
        raise ValueError(f"bitstring length {len(x)} != n_qubits {n}")

    # Build the computational basis state |x⟩
    # Qiskit Statevector: qubit 0 is LSB, so index = Σ_i x_i * 2^i
    idx = sum(int(x[q]) << q for q in range(n))
    sv_data = np.zeros(2**n, dtype=complex)
    sv_data[idx] = 1.0
    sv = Statevector(sv_data)
    return float(sv.expectation_value(hamiltonian).real)
