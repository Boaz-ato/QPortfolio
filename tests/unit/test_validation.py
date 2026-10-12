"""Unit tests for validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qportfolio.utils.validation import (
    DataValidationError,
    validate_array_finite,
    validate_date_range,
    validate_shape_consistency,
    validate_symmetric,
    validate_ticker_list,
)


def test_validate_ticker_list_passes():
    validate_ticker_list(["AAPL", "MSFT"])


def test_validate_ticker_list_empty():
    with pytest.raises(DataValidationError, match="empty"):
        validate_ticker_list([])


def test_validate_ticker_list_duplicates():
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_ticker_list(["AAPL", "MSFT", "AAPL"])


def test_validate_array_finite_passes():
    validate_array_finite(np.array([1.0, 2.0, 3.0]), "arr")


def test_validate_array_finite_nan():
    with pytest.raises(DataValidationError, match="NaN"):
        validate_array_finite(np.array([1.0, float("nan")]), "arr")


def test_validate_array_finite_inf():
    with pytest.raises(DataValidationError, match="Inf"):
        validate_array_finite(np.array([1.0, float("inf")]), "arr")


def test_validate_symmetric_passes():
    A = np.array([[1.0, 2.0], [2.0, 3.0]])
    validate_symmetric(A, "A")


def test_validate_symmetric_fails():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(DataValidationError, match="not symmetric"):
        validate_symmetric(A, "A")


def test_validate_symmetric_non_square():
    A = np.ones((2, 3))
    with pytest.raises(DataValidationError):
        validate_symmetric(A, "A")


def test_validate_date_range_passes():
    validate_date_range("2020-01-01", "2022-12-31")


def test_validate_date_range_start_after_end():
    with pytest.raises(ValueError, match="strictly before"):
        validate_date_range("2022-01-01", "2020-01-01")


def test_validate_date_range_same_date():
    with pytest.raises(ValueError):
        validate_date_range("2020-01-01", "2020-01-01")


def test_validate_date_range_bad_format():
    with pytest.raises(ValueError):
        validate_date_range("01/01/2020", "2022-12-31")


def test_validate_shape_consistency_passes():
    N = 4
    validate_shape_consistency(["A"] * N, np.ones(N), np.ones((N, N)), np.ones((N, N)))


def test_validate_shape_consistency_fails():
    with pytest.raises(ValueError, match="Shape inconsistencies"):
        validate_shape_consistency(["A", "B", "C"], np.ones(2), np.ones((3, 3)), np.ones((3, 3)))
