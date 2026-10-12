"""Unit tests for SimplexWeightSampler."""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.optimization.weight_sampling import SimplexWeightSampler


@pytest.fixture
def sampler():
    return SimplexWeightSampler(n_objectives=3)


def test_sample_shape(sampler):
    w = sampler.sample(50, seed=0)
    assert w.shape == (50, 3)


def test_sample_sums_to_one(sampler):
    w = sampler.sample(200, seed=42)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_sample_nonnegative(sampler):
    w = sampler.sample(200, seed=42)
    assert (w >= 0).all()


def test_sample_dtype_float64(sampler):
    w = sampler.sample(10, seed=0)
    assert w.dtype == np.float64


def test_sample_deterministic_with_seed(sampler):
    w1 = sampler.sample(100, seed=7)
    w2 = sampler.sample(100, seed=7)
    np.testing.assert_array_equal(w1, w2)


def test_sample_different_seeds_differ(sampler):
    w1 = sampler.sample(50, seed=1)
    w2 = sampler.sample(50, seed=2)
    assert not np.allclose(w1, w2)


def test_sample_one(sampler):
    rng = np.random.default_rng(0)
    w = sampler.sample_one(rng)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-12
    assert (w >= 0).all()


def test_sample_one_sequential(sampler):
    """Calling sample_one repeatedly on same rng produces different vectors."""
    rng = np.random.default_rng(99)
    w1 = sampler.sample_one(rng)
    w2 = sampler.sample_one(rng)
    assert not np.allclose(w1, w2)


def test_uniform_distribution():
    """Mean of many samples should be close to (1/3, 1/3, 1/3)."""
    sampler = SimplexWeightSampler(3)
    w = sampler.sample(10_000, seed=0)
    mean = w.mean(axis=0)
    np.testing.assert_allclose(mean, [1 / 3, 1 / 3, 1 / 3], atol=0.02)


def test_two_objective_sampler():
    sampler = SimplexWeightSampler(n_objectives=2)
    w = sampler.sample(100, seed=0)
    assert w.shape == (100, 2)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
