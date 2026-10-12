"""Base interface for QAOA parameter trainers.

A trainer receives a serialized Hamiltonian description and returns
optimized (β*, γ*) parameters for the QAOA circuit.

Concrete implementations:
  - Python COBYLA trainer (built into QAOASolver, no extra class needed)
  - JuliQAOATrainer   — subprocess bridge to Julia optimizer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingResult:
    """Outcome of a QAOA parameter training run.

    Parameters
    ----------
    params_gammas : (p,) trained γ parameters
    params_betas  : (p,) trained β parameters
    history       : per-iteration records (dicts with at least 'iter' and 'energy')
    n_iterations  : number of optimizer iterations performed
    final_energy  : –⟨H_C⟩ at the trained parameters (should be negative for maximization)
    training_time : wall-clock seconds for the training call
    converged     : whether the optimizer reported convergence
    backend       : identifier string for the training backend
    """

    params_gammas: np.ndarray
    params_betas: np.ndarray
    history: list[dict] = field(default_factory=list)
    n_iterations: int = 0
    final_energy: float = 0.0
    training_time: float = 0.0
    converged: bool = False
    backend: str = "unknown"

    def to_dict(self) -> dict:
        """Serializable summary for inclusion in OptimizationResult.quantum."""
        return {
            "backend": self.backend,
            "params_gammas": self.params_gammas.tolist(),
            "params_betas": self.params_betas.tolist(),
            "n_iterations": self.n_iterations,
            "final_energy": float(self.final_energy),
            "converged": bool(self.converged),
            "training_time": float(self.training_time),
        }


class BaseTrainer(ABC):
    """Abstract base for QAOA parameter trainers.

    Subclasses implement ``train()`` for a specific backend (Python/Julia/…).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. ``"cobyla_python"``)."""
        ...

    @abstractmethod
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
        """Optimize QAOA parameters and return a TrainingResult.

        Parameters
        ----------
        H_cost_pauli  : list of (pauli_string, coeff) for the cost Hamiltonian
        H_mixer_pauli : list of (pauli_string, coeff) for the mixer Hamiltonian
        n             : number of qubits
        k             : cardinality (Hamming weight of the Dicke initial state)
        p             : QAOA depth (number of layers)
        n_iters       : maximum optimizer iterations
        x0            : (2p,) initial parameter vector [γ_0…γ_{p-1}, β_0…β_{p-1}]
        seed          : random seed for stochastic optimizers
        """
        ...

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's external dependencies are available."""
        return True
