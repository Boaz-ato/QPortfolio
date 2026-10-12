"""Simplex weight sampler for multi-objective scalarization.

Samples weight vectors uniformly from the 3-dimensional probability simplex
using c ~ Dirichlet(1,1,1), which is equivalent to the uniform distribution on

  {(c1, c2, c3) : c1 + c2 + c3 = 1, c_j >= 0}.

Both WSM and QAOA use this sampler so comparisons use consistent weight vectors.
"""

from __future__ import annotations

import numpy as np


class SimplexWeightSampler:
    """Uniform sampler on the 3-objective probability simplex.

    Parameters
    ----------
    n_objectives:
        Number of objectives (default 3).
    """

    def __init__(self, n_objectives: int = 3) -> None:
        self.n_objectives = n_objectives

    def sample(self, n_weights: int, seed: int | None = None) -> np.ndarray:
        """Draw n_weights weight vectors from Dirichlet(1,...,1).

        Parameters
        ----------
        n_weights:
            Number of weight vectors to draw.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Shape (n_weights, n_objectives), each row sums to 1.
        """
        rng = np.random.default_rng(seed)
        # Dirichlet(alpha=1,...,1) = uniform on simplex
        weights = rng.dirichlet(np.ones(self.n_objectives), size=n_weights)
        return weights.astype(np.float64)

    def sample_one(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a single weight vector using an existing RNG.

        Parameters
        ----------
        rng:
            NumPy Generator (allows the caller to control the RNG state).

        Returns
        -------
        np.ndarray
            Shape (n_objectives,), sums to 1.
        """
        return rng.dirichlet(np.ones(self.n_objectives)).astype(np.float64)
