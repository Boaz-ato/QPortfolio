"""Hypervolume computation with normalization for fair cross-solver comparison."""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def _pymoo_hv(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """Compute raw hypervolume (maximization) using pymoo.

    Parameters
    ----------
    objectives : (M, D) array — Pareto-front objective vectors (maximization).
    reference_point : (D,) array — strictly dominated by all front points.

    Returns
    -------
    float — hypervolume contribution of objectives w.r.t. reference_point.
    """

    if len(objectives) == 0:
        return 0.0

    # pymoo uses minimization, so negate everything
    neg_ref = -reference_point
    ind = HV(ref_point=neg_ref)
    return float(ind(-objectives))


class NormalizedHypervolume:
    """Affine-normalizes objectives before HV computation for cross-problem comparison.

    Normalization maps each objective dimension via:
        f̃_j = (f_j - a_j) / s_j

    where ``a_j`` and ``s_j`` are fixed at construction time (problem-level scales)
    so that all algorithms are evaluated on the same scale.
    """

    def __init__(
        self,
        shift: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        shift : (D,) — subtracted from raw objectives before scaling.
        scale : (D,) — divided after shifting; must be > 0.
        """
        if np.any(scale <= 0):
            raise ValueError("All scale values must be positive.")
        self.shift = np.asarray(shift, dtype=float)
        self.scale = np.asarray(scale, dtype=float)

    def normalize(self, objectives: np.ndarray) -> np.ndarray:
        """Apply affine normalization.  Works for (M, D) or (D,) inputs."""
        return (objectives - self.shift) / self.scale

    def normalized_reference_point(self, reference_point: np.ndarray) -> np.ndarray:
        """Normalize a reference point with the same transform."""
        return self.normalize(reference_point)

    def compute(
        self,
        objectives: np.ndarray,
        reference_point: np.ndarray,
    ) -> float:
        """Normalized HV: normalize both front and reference before computing.

        Parameters
        ----------
        objectives : (M, D) — raw (un-normalized) objective vectors.
        reference_point : (D,) — raw reference point.
        """
        norm_objs = self.normalize(objectives)
        norm_ref = self.normalize(reference_point)
        return _pymoo_hv(norm_objs, norm_ref)

    def compute_raw(
        self,
        objectives: np.ndarray,
        reference_point: np.ndarray,
    ) -> float:
        """Raw (un-normalized) HV."""
        return _pymoo_hv(objectives, reference_point)

    @classmethod
    def from_reference_range(
        cls,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        margin: float = 0.1,
    ) -> NormalizedHypervolume:
        """Build normalizer from objective lower/upper bounds.

        Shift = lower_bounds, scale = upper_bounds - lower_bounds (with margin guard).
        """
        scale = upper_bounds - lower_bounds
        # Guard against zero-range objectives
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(shift=lower_bounds.copy(), scale=scale)


def compute_hypervolume(
    objectives: np.ndarray,
    reference_point: np.ndarray,
    normalizer: NormalizedHypervolume | None = None,
) -> float:
    """Convenience wrapper: raw or normalized HV depending on whether normalizer is given.

    Parameters
    ----------
    objectives : (M, D) array.
    reference_point : (D,) array.
    normalizer : if provided, normalized HV is returned; otherwise raw.
    """
    if normalizer is not None:
        return normalizer.compute(objectives, reference_point)
    return _pymoo_hv(objectives, reference_point)
