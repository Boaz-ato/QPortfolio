"""
qportfolio — Research library for classical and quantum multi-objective portfolio optimization.
"""

from qportfolio.data.covariance import CovarianceEstimator
from qportfolio.data.universe import StockUniverse
from qportfolio.data.yahoo import MarketDataProvider
from qportfolio.evaluation.benchmark import BenchmarkConfig, BenchmarkMode, BenchmarkSuite
from qportfolio.evaluation.hypervolume import NormalizedHypervolume, compute_hypervolume
from qportfolio.evaluation.reference import ReferenceStrategy, compute_reference_point
from qportfolio.evaluation.timing import SolutionDiscoveryEvent, SolverTimer
from qportfolio.optimization.pareto_archive import ParetoArchive
from qportfolio.optimization.weight_sampling import SimplexWeightSampler
from qportfolio.problems.constraints import ConstraintValidator
from qportfolio.problems.objectives import ObjectiveEvaluator
from qportfolio.problems.portfolio import PortfolioProblem
from qportfolio.solvers.epsilon import EpsilonCMSolver
from qportfolio.solvers.nsga2 import NSGA2Solver
from qportfolio.solvers.quantum.hamiltonians import HamiltonianComponents, build_hamiltonians
from qportfolio.solvers.quantum.mixers import (
    verify_xy_commutes_with_cardinality,
    xy_mixer_hamiltonian,
    xy_mixer_layer,
)
from qportfolio.solvers.quantum.qaoa import QAOASolver
from qportfolio.solvers.quantum.states.dicke import dicke_state_circuit, validate_dicke_circuit
from qportfolio.solvers.quantum.trainers import BaseTrainer, JuliQAOATrainer, TrainingResult
from qportfolio.solvers.random import RandomSampler
from qportfolio.solvers.wsm import WSMSolver

__version__ = "0.1.0"

__all__ = [
    "PortfolioProblem",
    "ObjectiveEvaluator",
    "ConstraintValidator",
    "MarketDataProvider",
    "StockUniverse",
    "CovarianceEstimator",
    "RandomSampler",
    "WSMSolver",
    "EpsilonCMSolver",
    "NSGA2Solver",
    "ParetoArchive",
    "SimplexWeightSampler",
    "ReferenceStrategy",
    "compute_reference_point",
    "NormalizedHypervolume",
    "compute_hypervolume",
    "BenchmarkSuite",
    "BenchmarkConfig",
    "BenchmarkMode",
    "SolverTimer",
    "SolutionDiscoveryEvent",
    "QAOASolver",
    "dicke_state_circuit",
    "validate_dicke_circuit",
    "build_hamiltonians",
    "HamiltonianComponents",
    "xy_mixer_layer",
    "verify_xy_commutes_with_cardinality",
    "xy_mixer_hamiltonian",
    "BaseTrainer",
    "TrainingResult",
    "JuliQAOATrainer",
]
