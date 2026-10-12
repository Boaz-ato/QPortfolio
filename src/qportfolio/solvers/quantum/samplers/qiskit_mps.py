"""Qiskit MPS (matrix product state) sampler for QAOA circuits.

Uses ``AerSimulator(method="matrix_product_state")`` to efficiently simulate
QAOA for moderately large N (up to ~30-40 qubits).

For each weight vector c^(m) from SimplexWeightSampler:
  1. Build H_C(c^(m)) from HamiltonianComponents
  2. Bind trained (β*, γ*) parameters
  3. Sample S shots
  4. Validate Hamming weight of every bitstring — reject non-K strings
  5. Add feasible (Hamming-K) strings to the Pareto archive

Tracks:
  - Total measurement shots, unique bitstrings, unique feasible bitstrings
  - MPS bond dimension
  - Circuit metrics pre/post transpilation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from tqdm import tqdm

from qportfolio.optimization.pareto_archive import ParetoArchive
from qportfolio.solvers.quantum.hamiltonians import HamiltonianComponents
from qportfolio.solvers.quantum.mixers import xy_mixer_layer


@dataclass
class MPSSamplerConfig:
    """Configuration for the Qiskit MPS sampler."""

    shots: int = 1024
    """Shots per weight vector."""

    bond_dimension: int = 32
    """MPS truncation bond dimension (larger = more accurate, slower)."""

    truncation_threshold: float = 1e-6
    """MPS singular value truncation threshold."""

    seed: int = 0
    """Simulator seed for reproducibility."""


@dataclass
class SamplingStats:
    """Aggregate statistics from one sampling run (all weight vectors)."""

    n_weight_vectors: int = 0
    total_shots: int = 0
    unique_bitstrings: int = 0
    unique_feasible: int = 0
    n_nondominated: int = 0
    sampling_time: float = 0.0
    hardware_violations: int = 0
    """Bitstrings with wrong Hamming weight (should be 0 for XY mixer)."""


def _build_qaoa_circuit(
    n: int,
    p: int,
    initial_circuit,
    cost_circuit_fn,
    mixer_circuit_fn,
) -> tuple:
    """Assemble the full QAOA circuit: initial state + p layers of (cost, mixer).

    Parameters
    ----------
    n : number of qubits
    p : QAOA depth
    initial_circuit : QuantumCircuit — the initial state (Dicke)
    cost_circuit_fn : callable(gamma) → QuantumCircuit — one cost unitary layer
    mixer_circuit_fn : callable(beta) → QuantumCircuit — one mixer layer

    Returns
    -------
    Tuple of (QuantumCircuit, gammas ParameterVector, betas ParameterVector)
    with 2p parameters (γ_0...γ_{p-1}, β_0...β_{p-1})
    """

    gammas = ParameterVector("γ", p)
    betas = ParameterVector("β", p)

    qc = QuantumCircuit(n, name=f"QAOA(p={p})")
    qc.compose(initial_circuit, inplace=True)

    for layer in range(p):
        qc.barrier()
        qc.compose(cost_circuit_fn(gammas[layer]), inplace=True)
        qc.barrier()
        qc.compose(mixer_circuit_fn(betas[layer]), inplace=True)

    return qc, gammas, betas


def _cost_unitary_from_hamiltonian(
    hamiltonian,
    gamma,
    n: int,
):
    """Build e^{-i γ H_C} via PauliEvolutionGate (Trotterized).

    This uses the built-in Qiskit synthesis which maps the SparsePauliOp
    to a product of Pauli rotations (exact for single-layer Trotter).
    """

    synth = LieTrotter(reps=1)
    gate = PauliEvolutionGate(hamiltonian, time=gamma, synthesis=synth)
    qc = QuantumCircuit(n, name="cost")
    qc.append(gate, range(n))
    return qc.decompose()


class QiskitMPSSampler:
    """Sample QAOA circuits using Aer's MPS simulator.

    Parameters
    ----------
    hamiltonian_components : HamiltonianComponents (built from the problem)
    config : MPSSamplerConfig
    """

    def __init__(
        self,
        hamiltonian_components: HamiltonianComponents,
        config: MPSSamplerConfig | None = None,
    ) -> None:
        self.hc = hamiltonian_components
        self.config = config or MPSSamplerConfig()
        self._simulator = None

    def _get_simulator(self):
        if self._simulator is None:
            self._simulator = AerSimulator(
                method="matrix_product_state",
                matrix_product_state_max_bond_dimension=self.config.bond_dimension,
                matrix_product_state_truncation_threshold=self.config.truncation_threshold,
                seed_simulator=self.config.seed,
            )
        return self._simulator

    def sample(
        self,
        weight_vectors: np.ndarray,
        params_gammas: np.ndarray,
        params_betas: np.ndarray,
        archive: ParetoArchive,
        initial_circuit,
        mixer_topology: str = "ring",
        p: int = 1,
        elapsed_start: float = 0.0,
    ) -> SamplingStats:
        """Run QAOA sampling for all weight vectors and add to archive.

        Parameters
        ----------
        weight_vectors : (M, 3) weight matrix
        params_gammas : (p,) trained γ parameters
        params_betas : (p,) trained β parameters
        archive : ParetoArchive to add solutions to
        initial_circuit : Dicke state circuit
        mixer_topology : XY mixer topology
        p : QAOA circuit depth
        elapsed_start : time already elapsed before this call (for archive events)
        """

        n = self.hc.n_qubits
        stats = SamplingStats(n_weight_vectors=len(weight_vectors))
        t0 = time.perf_counter()

        sim = self._get_simulator()
        seen_bitstrings: set[bytes] = set()
        M = len(weight_vectors)

        # --- Build all circuits first, then batch-transpile and batch-run ---
        t_build = time.perf_counter()

        bound_circuits = []
        for weights in tqdm(
            weight_vectors, desc="[QAOA] Building circuits", unit="circuit", leave=False
        ):
            H_w = self.hc.weighted(weights)

            def _cost(gamma, _H=H_w):
                return _cost_unitary_from_hamiltonian(_H, gamma, n)

            def _mix(beta):
                return xy_mixer_layer(n, beta, mixer_topology)

            qaoa_qc, gammas_pv, betas_pv = _build_qaoa_circuit(n, p, initial_circuit.copy(), _cost, _mix)
            qaoa_qc.measure_all()

            param_dict = {}
            for layer in range(p):
                param_dict[gammas_pv[layer]] = float(params_gammas[layer])
                param_dict[betas_pv[layer]] = float(params_betas[layer])
            bound_circuits.append(qaoa_qc.assign_parameters(param_dict))

        # Batch transpile — one call for all circuits.
        # transpile decomposes PauliEvolutionGate (via its LieTrotter synthesis)
        # and all other composite gates in a single pass.
        transpiled = transpile(bound_circuits, sim, optimization_level=0)

        tqdm.write(
            f"[QAOA]   Build+transpile done in {time.perf_counter() - t_build:.1f}s  "
            f"(depth={transpiled[0].depth()}, gates={transpiled[0].size()})"
        )

        # Run in batches of 32 to bound per-job memory usage.
        BATCH_SIZE = 32
        all_counts = []
        n_batches = (M + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_start in tqdm(
            range(0, M, BATCH_SIZE),
            total=n_batches,
            desc="[QAOA] Running batches",
            unit="batch",
            leave=False,
        ):
            batch = transpiled[batch_start : batch_start + BATCH_SIZE]
            job = sim.run(batch, shots=self.config.shots)
            counts = job.result().get_counts()
            if isinstance(counts, dict):
                counts = [counts]
            all_counts.extend(counts)

        stats.total_shots += self.config.shots * M

        prev_nondominated = 0

        result_bar = tqdm(
            zip(weight_vectors, all_counts),
            total=M,
            desc="[QAOA] Processing results",
            unit="weight",
            leave=False,
        )
        for weights, counts in result_bar:
            for bitstring, count in counts.items():
                # Qiskit bitstrings are big-endian (q_{n-1}...q_0).
                # Convert to little-endian array: x[i] = bit for qubit i.
                bits_le = np.array(list(reversed(bitstring)), dtype=float)
                hw = int(bits_le.sum())

                if hw != archive._problem.K:
                    stats.hardware_violations += count
                    continue

                key = bits_le.astype(np.uint8).tobytes()
                if key not in seen_bitstrings:
                    seen_bitstrings.add(key)
                    stats.unique_bitstrings += 1
                    stats.unique_feasible += 1

                objs = np.array(archive._problem.evaluate(bits_le))
                elapsed = elapsed_start + (time.perf_counter() - t0)
                archive.add(bits_le, objs, elapsed_time=elapsed, source="qaoa_mps")
                if archive.n_nondominated > prev_nondominated:
                    prev_nondominated = archive.n_nondominated
                    tqdm.write(
                        f"[QAOA]   >> new non-dominated solution  "
                        f"(total={prev_nondominated})  "
                        f"f=({objs[0]:.4f}, {objs[1]:.4f}, {objs[2]:.4f})"
                    )

            result_bar.set_postfix(
                feasible=stats.unique_feasible,
                nd=archive.n_nondominated,
            )

        stats.sampling_time = time.perf_counter() - t0
        stats.n_nondominated = archive.n_nondominated
        return stats
