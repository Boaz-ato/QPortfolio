"""QAOA solver for multi-objective portfolio optimization.

Circuit:
  - Initial state: |D_K^N⟩ (uniform Dicke state)
  - Cost unitary:  e^{-i γ H_C(c)} where H_C(c) = c^T (H_risk, H_return, H_diversif)
  - Mixer unitary: e^{-i β H_M}  where H_M is the XY mixer

Training strategy:
  - Train once on an equal-weight vector c_train = (1/3, 1/3, 1/3).
  - Transfer the trained (β*, γ*) to all M weight vectors in the sweep.

Runtime decomposition (QAOATiming):
  - T_cold     = T_training + T_sampling + T_overhead
  - T_amortized = T_training/R + T_sampling + T_overhead  (R = n_repetitions = 1)
"""

from __future__ import annotations

import time

import numpy as np
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

from qportfolio.evaluation.timing import QAOATiming
from qportfolio.optimization.pareto_archive import ParetoArchive
from qportfolio.optimization.weight_sampling import SimplexWeightSampler
from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.solvers.base import BaseSolver, OptimizationResult
from qportfolio.solvers.quantum.hamiltonians import build_hamiltonians
from qportfolio.solvers.quantum.mixers import xy_mixer_hamiltonian, xy_mixer_layer
from qportfolio.solvers.quantum.samplers.qiskit_mps import (
    MPSSamplerConfig,
    QiskitMPSSampler,
    _build_qaoa_circuit,
    _cost_unitary_from_hamiltonian,
)
from qportfolio.solvers.quantum.states.dicke import dicke_state_circuit
from qportfolio.solvers.quantum.trainers.base import BaseTrainer, TrainingResult
from qportfolio.solvers.quantum.trainers.juliqaoa import JuliQAOATrainer


class QAOASolver(BaseSolver):
    """Multi-objective QAOA solver with uniform Dicke initial state.

    Parameters
    ----------
    p : int
        QAOA circuit depth (layers).
    n_weights : int
        Number of weight vectors to sample for the objective sweep.
    shots : int
        Measurement shots per weight vector.
    seed : int
        Master RNG seed (used for weights, training, simulator).
    mixer_topology : str
        XY mixer topology: ``"ring"``, ``"chain"``, or ``"complete"``.
    bond_dimension : int
        MPS bond dimension for Aer simulator.
    training_shots : int
        Shots used during VQE-style training (per cost-function evaluation).
    n_training_iters : int
        Number of gradient-free optimizer iterations for parameter training.
    hv_reference_point : np.ndarray, optional
        Reference point for hypervolume tracking.
    trainer : BaseTrainer, optional
        External parameter trainer (e.g. ``JuliQAOATrainer``).  When ``None``
        (default) the built-in Python COBYLA trainer is used.
    """

    algorithm_name = "qaoa"

    # Maximum problem size for direct statevector training in Julia.
    # Dense matrices are 2^n × 2^n complex: n=12 → 256 MB (feasible),
    # n=15 → 17 GB (infeasible). Problems with N > PROXY_N_MAX train on
    # a PROXY_N_MAX-qubit proxy and transfer β*, γ* to the full circuit.
    PROXY_N_MAX: int = 12

    def __init__(
        self,
        p: int = 1,
        n_weights: int = 50,
        shots: int = 1024,
        seed: int = 0,
        mixer_topology: str = "ring",
        bond_dimension: int = 32,
        training_shots: int = 512,
        n_training_iters: int = 100,
        hv_reference_point: np.ndarray | None = None,
        trainer: BaseTrainer | None = None,
    ) -> None:
        self.p = p
        self.n_weights = n_weights
        self.shots = shots
        self.seed = seed
        self.mixer_topology = mixer_topology
        self.bond_dimension = bond_dimension
        self.training_shots = training_shots
        self.n_training_iters = n_training_iters
        self.hv_reference_point = hv_reference_point
        self.trainer = trainer

    def solve(self, problem: PortfolioProblem) -> OptimizationResult:
        """Run QAOA on the given portfolio problem.

        Returns
        -------
        OptimizationResult with algorithm="qaoa".
        """

        wall_start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        timing = QAOATiming()

        N = len(problem.tickers)
        print(
            f"[QAOA] Starting solver  N={N}, K={problem.K}, p={self.p}, "
            f"n_weights={self.n_weights}, shots={self.shots}"
        )

        archive = ParetoArchive(problem, hv_cadence=10)
        if self.hv_reference_point is not None:
            archive.set_reference_point(self.hv_reference_point)

        # ------------------------------------------------------------------
        # Build problem-level objects
        # ------------------------------------------------------------------
        print("[QAOA] Building Hamiltonians (risk, return, diversification)...")
        hc = build_hamiltonians(problem.mu, problem.sigma, problem.D)
        c_train = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        # ------------------------------------------------------------------
        # Initial state: uniform Dicke |D_K^N⟩
        # ------------------------------------------------------------------
        print(f"[QAOA] Preparing uniform Dicke initial state |D_{N}^{problem.K}>...")
        initial_qc, _ = dicke_state_circuit(n=N, k=problem.K)

        # ------------------------------------------------------------------
        # Training: find (β*, γ*) using the equal-weight Hamiltonian
        # ------------------------------------------------------------------
        t_train_start = time.perf_counter()

        # Resolve trainer: explicit > auto-detect Julia
        active_trainer = self.trainer
        if active_trainer is None:
            if JuliQAOATrainer.is_available():
                active_trainer = JuliQAOATrainer(mixer_topology=self.mixer_topology)

        if active_trainer is not None:
            backend_name = getattr(active_trainer, "name", type(active_trainer).__name__)
            if N > self.PROXY_N_MAX:
                n_proxy = self.PROXY_N_MAX
                k_proxy = min(problem.K, n_proxy - 1)
                print(
                    f"[QAOA] N={N} > {self.PROXY_N_MAX}: training on {n_proxy}-qubit proxy "
                    f"(K={k_proxy}) with {backend_name} (max_iters={self.n_training_iters})..."
                )
            else:
                n_proxy = N
                k_proxy = problem.K
                print(
                    f"[QAOA] Training parameters with {backend_name} "
                    f"(max_iters={self.n_training_iters})..."
                )
            train_result = self._train_external(
                hc=hc,
                c_train=c_train,
                rng=rng,
                problem=problem,
                trainer=active_trainer,
                n_proxy=n_proxy,
                k_proxy=k_proxy,
            )
            params_gammas = train_result.params_gammas
            params_betas = train_result.params_betas
            active_trainer_name = backend_name
        else:
            print(f"[QAOA] Training parameters with COBYLA (max_iters={self.n_training_iters})...")
            params_gammas, params_betas = self._train(
                hc=hc,
                c_train=c_train,
                dicke_qc=initial_qc,
                rng=rng,
                problem=problem,
            )
            train_result = None
            active_trainer_name = "cobyla_python"
        timing.training_time = time.perf_counter() - t_train_start

        g_str = ", ".join(f"{g:.4f}" for g in params_gammas)
        b_str = ", ".join(f"{b:.4f}" for b in params_betas)
        print(f"[QAOA] Training done in {timing.training_time:.2f}s  γ*=[{g_str}]  β*=[{b_str}]")

        # ------------------------------------------------------------------
        # Sampling sweep: M weight vectors
        # ------------------------------------------------------------------
        print(
            f"[QAOA] Sampling sweep: {self.n_weights} weight vectors × "
            f"{self.shots} shots = {self.n_weights * self.shots} total shots..."
        )
        t_sample_start = time.perf_counter()
        weight_sampler = SimplexWeightSampler(n_objectives=3)
        weight_vectors = weight_sampler.sample(self.n_weights, seed=self.seed)

        sampler_cfg = MPSSamplerConfig(
            shots=self.shots,
            bond_dimension=self.bond_dimension,
            seed=int(rng.integers(0, 2**31)),
        )
        mps_sampler = QiskitMPSSampler(hc, sampler_cfg)
        stats = mps_sampler.sample(
            weight_vectors=weight_vectors,
            params_gammas=params_gammas,
            params_betas=params_betas,
            archive=archive,
            initial_circuit=initial_qc,
            mixer_topology=self.mixer_topology,
            p=self.p,
            elapsed_start=timing.training_time,
        )
        timing.sampling_time = time.perf_counter() - t_sample_start

        print(
            f"[QAOA] Sampling done in {timing.sampling_time:.2f}s  "
            f"unique_feasible={stats.unique_feasible}, "
            f"hw_violations={stats.hardware_violations}, "
            f"non_dominated={archive.n_nondominated}"
        )

        elapsed = time.perf_counter() - wall_start
        timing.other_time = elapsed - timing.training_time - timing.sampling_time

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        total_runtime = time.perf_counter() - wall_start
        quantum_meta = {
            "p": self.p,
            "n_weights": self.n_weights,
            "shots_per_weight": self.shots,
            "mixer_topology": self.mixer_topology,
            "bond_dimension": self.bond_dimension,
            "n_training_iters": self.n_training_iters,
            "params_gammas": params_gammas.tolist(),
            "params_betas": params_betas.tolist(),
            "training_weight": c_train.tolist(),
            "trainer": active_trainer_name,
            "trainer_result": train_result.to_dict() if train_result is not None else None,
            "total_shots": stats.total_shots,
            "unique_feasible": stats.unique_feasible,
            "hardware_violations": stats.hardware_violations,
            "timing": timing.to_dict(),
        }

        return self._make_result(
            archive=archive,
            algorithm="qaoa",
            problem=problem,
            runtime=total_runtime,
            n_evaluations=stats.unique_feasible,
            status="feasible" if archive.n_nondominated > 0 else "failed",
            quantum=quantum_meta,
        )

    # ------------------------------------------------------------------
    # External trainer dispatch
    # ------------------------------------------------------------------

    def _train_external(
        self,
        hc,
        c_train: np.ndarray,
        rng: np.random.Generator,
        problem: PortfolioProblem,
        trainer: BaseTrainer | None = None,
        n_proxy: int | None = None,
        k_proxy: int | None = None,
    ) -> TrainingResult:
        """Delegate training to a trainer (e.g. JuliQAOATrainer).

        When n_proxy < hc.n_qubits, builds proxy Hamiltonians from the first
        n_proxy stocks and trains on those, then transfers (β*, γ*) to the full
        N-qubit sampling sweep.
        """

        active = trainer if trainer is not None else self.trainer
        assert active is not None, "_train_external requires a non-None trainer"

        if n_proxy is not None and n_proxy < hc.n_qubits:
            # Build proxy Hamiltonians from the first n_proxy stocks
            hc_proxy = build_hamiltonians(
                problem.mu[:n_proxy],
                problem.sigma[:n_proxy, :n_proxy],
                problem.D[:n_proxy, :n_proxy],
            )
            H_cost_op = hc_proxy.weighted(c_train)
            H_mixer_op = xy_mixer_hamiltonian(n_proxy, self.mixer_topology)
            train_n = n_proxy
            train_k = k_proxy if k_proxy is not None else min(problem.K, n_proxy - 1)
        else:
            H_cost_op = hc.weighted(c_train)
            H_mixer_op = xy_mixer_hamiltonian(hc.n_qubits, self.mixer_topology)
            train_n = hc.n_qubits
            train_k = problem.K

        cost_pauli = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_cost_op)
        mixer_pauli = JuliQAOATrainer.hamiltonian_to_pauli_terms(H_mixer_op)

        x0 = rng.uniform(-np.pi, np.pi, 2 * self.p)

        return active.train(
            H_cost_pauli=cost_pauli,
            H_mixer_pauli=mixer_pauli,
            n=train_n,
            k=train_k,
            p=self.p,
            n_iters=self.n_training_iters,
            x0=x0,
            seed=int(rng.integers(0, 2**31)),
        )

    # ------------------------------------------------------------------
    # Internal training (Python / COBYLA fallback)
    # ------------------------------------------------------------------

    def _train(
        self,
        hc,
        c_train: np.ndarray,
        dicke_qc,
        rng: np.random.Generator,
        problem: PortfolioProblem,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find (β*, γ*) by maximizing ⟨H_C(c_train)⟩ using COBYLA.

        Returns
        -------
        (params_gammas, params_betas) — each of shape (p,)
        """

        H_w = hc.weighted(c_train)
        n = hc.n_qubits

        def _cost(gamma):
            return _cost_unitary_from_hamiltonian(H_w, gamma, n)

        def _mix(beta):
            return xy_mixer_layer(n, beta, self.mixer_topology)

        qaoa_qc, gammas_pv, betas_pv = _build_qaoa_circuit(n, self.p, dicke_qc.copy(), _cost, _mix)

        estimator = StatevectorEstimator()

        def objective(params_flat: np.ndarray) -> float:
            """Return -⟨H_C⟩ (we minimize)."""
            gammas = params_flat[: self.p]
            betas = params_flat[self.p :]
            param_dict = {}
            for layer in range(self.p):
                param_dict[gammas_pv[layer]] = float(gammas[layer])
                param_dict[betas_pv[layer]] = float(betas[layer])
            bound = qaoa_qc.assign_parameters(param_dict)
            pub = (bound, H_w)
            result = estimator.run([pub]).result()
            return -float(result[0].data.evs)

        x0 = rng.uniform(-np.pi, np.pi, 2 * self.p)

        opt = minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": self.n_training_iters, "rhobeg": 0.5},
        )

        params_gammas = opt.x[: self.p]
        params_betas = opt.x[self.p :]
        return params_gammas, params_betas
