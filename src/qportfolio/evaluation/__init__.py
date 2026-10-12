"""Evaluation engine: reference points, hypervolume, benchmarking, timing."""

from qportfolio.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkSuite,
    ComparisonTable,
    SolverBenchmarkResult,
)
from qportfolio.evaluation.hypervolume import (
    NormalizedHypervolume,
    compute_hypervolume,
)
from qportfolio.evaluation.reference import (
    ReferenceStrategy,
    build_normalizer,
    compute_reference_point,
)
from qportfolio.evaluation.timing import QAOATiming, SolutionDiscoveryEvent, SolverTimer

__all__ = [
    "SolverTimer",
    "SolutionDiscoveryEvent",
    "QAOATiming",
    "NormalizedHypervolume",
    "compute_hypervolume",
    "ReferenceStrategy",
    "compute_reference_point",
    "build_normalizer",
    "BenchmarkConfig",
    "BenchmarkMode",
    "BenchmarkSuite",
    "ComparisonTable",
    "SolverBenchmarkResult",
]
