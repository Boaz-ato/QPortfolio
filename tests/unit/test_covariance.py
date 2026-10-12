"""Unit tests for CovarianceEstimator."""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.config import CovarianceConfig
from qportfolio.data.covariance import CovarianceEstimator
from qportfolio.data.preprocessing import compute_daily_returns


@pytest.fixture
def returns_df(tiny_prices_df):
    return compute_daily_returns(tiny_prices_df)


@pytest.fixture
def full_estimator(returns_df):
    est = CovarianceEstimator(CovarianceConfig(method="full"))
    est.fit(returns_df)
    return est


# ---------------------------------------------------------------------------
# Full covariance
# ---------------------------------------------------------------------------


def test_full_covariance_is_symmetric(full_estimator):
    sigma = full_estimator.sigma_full_
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)


def test_full_covariance_is_psd(full_estimator):
    is_psd, min_eig = full_estimator.check_psd(full_estimator.sigma_full_)
    assert is_psd, f"Full covariance not PSD: min eigenvalue = {min_eig}"


def test_full_covariance_shape(full_estimator, tiny_prices_df):
    N = len(tiny_prices_df.columns)
    assert full_estimator.sigma_full_.shape == (N, N)


def test_full_estimator_no_sparse(full_estimator):
    assert full_estimator.sigma_sparse_ is None
    assert full_estimator.threshold_ is None


def test_unfitted_estimator_raises():
    est = CovarianceEstimator()
    with pytest.raises(RuntimeError, match="fit"):
        _ = est.sigma_full_


# ---------------------------------------------------------------------------
# Sparsify
# ---------------------------------------------------------------------------


def test_sparsify_preserves_diagonal(full_estimator):
    sigma = full_estimator.sigma_full_
    for tau in [0.1, 0.3, 0.5, 0.8]:
        sparse = full_estimator.sparsify(sigma, tau)
        np.testing.assert_allclose(np.diag(sparse), np.diag(sigma), atol=1e-12)


def test_sparsify_zeros_low_correlation():
    """Construct sigma where rho_12 = 0.1. With tau=0.2: sparse[0,1] == 0."""
    sigma_1 = 0.04  # var of asset 0
    sigma_2 = 0.09  # var of asset 1
    rho_12 = 0.1
    cov_12 = rho_12 * np.sqrt(sigma_1) * np.sqrt(sigma_2)
    sigma = np.array([[sigma_1, cov_12], [cov_12, sigma_2]])

    est = CovarianceEstimator()
    sparse = est.sparsify(sigma, threshold=0.2)
    assert sparse[0, 1] == 0.0
    assert sparse[1, 0] == 0.0
    assert sparse[0, 0] == sigma_1
    assert sparse[1, 1] == sigma_2


def test_sparsify_keeps_high_correlation():
    """Construct sigma where rho_12 = 0.9. With tau=0.5: off-diagonal is kept."""
    sigma_1 = 0.04
    sigma_2 = 0.09
    rho_12 = 0.9
    cov_12 = rho_12 * np.sqrt(sigma_1) * np.sqrt(sigma_2)
    sigma = np.array([[sigma_1, cov_12], [cov_12, sigma_2]])

    est = CovarianceEstimator()
    sparse = est.sparsify(sigma, threshold=0.5)
    assert abs(sparse[0, 1] - cov_12) < 1e-12
    assert abs(sparse[1, 0] - cov_12) < 1e-12


def test_sparsify_zero_threshold_returns_copy(full_estimator):
    sigma = full_estimator.sigma_full_
    sparse = full_estimator.sparsify(sigma, 0.0)
    np.testing.assert_array_equal(sigma, sparse)
    assert sparse is not sigma  # must be a copy


def test_sparsify_does_not_mutate_input(full_estimator):
    sigma = full_estimator.sigma_full_.copy()
    original = sigma.copy()
    full_estimator.sparsify(sigma, 0.3)
    np.testing.assert_array_equal(sigma, original)


# ---------------------------------------------------------------------------
# PSD check and diagonal loading
# ---------------------------------------------------------------------------


def test_check_psd_returns_true_for_psd(full_estimator):
    sigma = full_estimator.sigma_full_
    is_psd, _ = full_estimator.check_psd(sigma)
    assert is_psd


def test_check_psd_returns_false_for_indefinite():
    # Construct a matrix with a negative eigenvalue
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
    est = CovarianceEstimator()
    is_psd, min_eig = est.check_psd(sigma)
    assert is_psd is False
    assert min_eig < 0


def test_diagonal_loading_makes_psd():
    """Apply loading to indefinite matrix → result is PSD."""
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])
    est = CovarianceEstimator()
    result, lam = est.apply_diagonal_loading(sigma)
    is_psd, min_eig = est.check_psd(result)
    assert is_psd, f"After loading: min_eig={min_eig}"
    assert lam > 0


def test_diagonal_loading_custom_lam():
    sigma = np.eye(3)
    est = CovarianceEstimator()
    result, lam = est.apply_diagonal_loading(sigma, lam=0.5)
    assert lam == 0.5
    np.testing.assert_allclose(np.diag(result), [1.5, 1.5, 1.5])


# ---------------------------------------------------------------------------
# Sparse estimation
# ---------------------------------------------------------------------------


def test_sparse_estimator_threshold_auto(returns_df):
    cfg = CovarianceConfig(method="sparse", threshold="auto", max_risk_error=0.5)
    est = CovarianceEstimator(cfg)
    est.fit(returns_df)
    assert est.sigma_sparse_ is not None
    assert est.threshold_ is not None
    assert 0.0 <= est.threshold_ < 1.0


def test_sparse_estimator_preserves_diagonal(returns_df):
    cfg = CovarianceConfig(method="sparse", threshold=0.1)
    est = CovarianceEstimator(cfg)
    est.fit(returns_df)
    np.testing.assert_allclose(
        np.diag(est.sigma_sparse_),
        np.diag(est.sigma_full_),
        atol=1e-12,
    )


def test_sparsification_report_completeness(returns_df):
    cfg = CovarianceConfig(method="sparse", threshold=0.1)
    est = CovarianceEstimator(cfg)
    est.fit(returns_df)
    report = est.sparsification_report_
    required_keys = {
        "threshold",
        "retained_fraction",
        "psd_before_loading",
        "psd_after_loading",
        "diagonal_loading_applied",
        "lam_used",
        "min_eigenvalue_before",
        "min_eigenvalue_after",
    }
    assert required_keys.issubset(set(report.keys())), (
        f"Missing keys: {required_keys - set(report.keys())}"
    )


def test_retained_fraction_decreases_with_threshold(returns_df):
    """Higher threshold → fewer retained off-diagonal entries."""
    cfg_low = CovarianceConfig(method="sparse", threshold=0.1)
    cfg_high = CovarianceConfig(method="sparse", threshold=0.5)
    est_low = CovarianceEstimator(cfg_low).fit(returns_df)
    est_high = CovarianceEstimator(cfg_high).fit(returns_df)
    assert est_low.retained_fraction_ >= est_high.retained_fraction_
