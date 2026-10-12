"""Julia-based QAOA parameter trainer (statevector backend).

Bridges Python to a self-contained Julia script (``julia/qaoa_train.jl``) via
subprocess + JSON serialization.

The Julia script implements statevector QAOA using Julia stdlib
(LinearAlgebra) plus Optim.jl (auto-installed on first run).

Workflow
--------
1. Serialize the cost and mixer Hamiltonians as Pauli-term lists (JSON).
2. Spawn ``julia <path>/qaoa_train.jl <input.json> <output.json>``.
3. Parse the output JSON into a ``TrainingResult``.

The Julia script is located at::

    src/qportfolio/solvers/quantum/julia/qaoa_train.jl

relative to this file.

Note: Training is limited to n ≤ 12 qubits. Dense matrices are 2^n × 2^n
complex: n=12 → ~256 MB (feasible), n=15 → ~17 GB (infeasible).
For larger problems, ``QAOASolver`` automatically trains on a 12-qubit proxy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from qportfolio.solvers.quantum.trainers.base import BaseTrainer, TrainingResult

# Path to the bundled Julia script
_JULIA_DIR = Path(__file__).parent.parent / "julia"
_JULIA_SCRIPT = _JULIA_DIR / "qaoa_train.jl"


class JuliQAOATrainer(BaseTrainer):
    """QAOA parameter optimizer backed by a Julia subprocess.

    Parameters
    ----------
    n_iters : int
        Maximum Optim.jl iterations (passed to Julia).
    optimizer : str
        Julia Optim.jl optimizer name: ``"BFGS"`` or ``"NelderMead"``.
        BFGS uses finite-difference gradients and converges faster for p≤3.
    julia_executable : str
        Path or name of the ``julia`` binary.  Defaults to ``"julia"``
        (resolved via PATH).
    timeout : float | None
        Wall-clock timeout for the Julia process in seconds.  ``None`` = no limit.
    """

    @property
    def name(self) -> str:
        return f"juliqaoa_{self.optimizer.lower()}_sv"

    def __init__(
        self,
        n_iters: int = 200,
        optimizer: str = "BFGS",
        julia_executable: str = "julia",
        timeout: float | None = None,
        mixer_topology: str = "ring",
    ) -> None:
        if optimizer not in ("BFGS", "NelderMead"):
            raise ValueError(f"optimizer must be 'BFGS' or 'NelderMead', got {optimizer!r}")
        self.n_iters = n_iters
        self.optimizer = optimizer
        self.julia_executable = julia_executable
        self.timeout = timeout
        self.mixer_topology = mixer_topology

    @staticmethod
    def is_available(julia_executable: str = "julia") -> bool:
        """Return True if ``julia`` is on PATH and the training script exists."""
        return shutil.which(julia_executable) is not None and _JULIA_SCRIPT.exists()

    def train(
        self,
        H_cost_pauli: list[tuple[str, complex]],
        H_mixer_pauli: list[tuple[str, complex]],
        n: int,
        k: int,
        p: int,
        n_iters: int,
        x0: np.ndarray,
        seed: int = 0,
    ) -> TrainingResult:
        """Run QAOA parameter training in Julia and return a ``TrainingResult``.

        Raises
        ------
        RuntimeError
            If Julia is not found, the script errors, or the output is malformed.
        """
        if not _JULIA_SCRIPT.exists():
            raise RuntimeError(
                f"Julia training script not found at {_JULIA_SCRIPT}. "
                "Ensure the package is installed correctly."
            )

        input_payload = {
            "pauli_terms_cost": [
                {"label": label, "coeff_re": float(c.real), "coeff_im": float(c.imag)}
                for label, c in H_cost_pauli
            ],
            "pauli_terms_mixer": [
                {"label": label, "coeff_re": float(c.real), "coeff_im": float(c.imag)}
                for label, c in H_mixer_pauli
            ],
            "n": int(n),
            "k": int(k),
            "p": int(p),
            "n_iters": int(n_iters),
            "x0": [float(v) for v in x0],
            "seed": int(seed),
            "optimizer": self.optimizer,
            "mixer_topology": self.mixer_topology,
        }

        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "qaoa_input.json")
            output_path = os.path.join(tmpdir, "qaoa_output.json")

            with open(input_path, "w") as f:
                json.dump(input_payload, f)

            cmd = [self.julia_executable, str(_JULIA_SCRIPT), input_path, output_path]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Julia executable not found: {self.julia_executable!r}. "
                    "Install Julia and ensure it is on PATH."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Julia QAOA training timed out after {self.timeout}s.") from exc

            if proc.stderr.strip():
                print(proc.stderr.strip(), flush=True)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Julia training script exited with code {proc.returncode}.\n"
                    f"--- stdout ---\n{proc.stdout}\n"
                    f"--- stderr ---\n{proc.stderr}"
                )

            if not os.path.exists(output_path):
                raise RuntimeError(
                    "Julia training script did not produce output file.\n"
                    f"--- stdout ---\n{proc.stdout}\n"
                    f"--- stderr ---\n{proc.stderr}"
                )

            with open(output_path) as f:
                result_dict = json.load(f)

        training_time = time.perf_counter() - t0

        return TrainingResult(
            params_gammas=np.array(result_dict["params_gammas"], dtype=float),
            params_betas=np.array(result_dict["params_betas"], dtype=float),
            history=result_dict.get("history", []),
            n_iterations=int(result_dict.get("n_iterations", 0)),
            final_energy=float(result_dict.get("final_energy", 0.0)),
            training_time=training_time,
            converged=bool(result_dict.get("converged", False)),
            backend=self.name,
        )

    # ------------------------------------------------------------------
    # Convenience: build Pauli-term lists from qportfolio objects
    # ------------------------------------------------------------------

    @staticmethod
    def hamiltonian_to_pauli_terms(op) -> list[tuple[str, complex]]:
        """Convert a Qiskit SparsePauliOp to a list of (label, coeff) pairs."""
        terms = []
        for pauli, coeff in zip(op.paulis, op.coeffs):
            label = str(pauli)
            terms.append((label, complex(coeff)))
        return terms
