"""Regression tests for PortfolioProblem serialization schema versioning.

Verifies that:
1. Saved problems embed schema_version="1.0" in problem.json.
2. Loading a problem with an unknown schema version raises ValueError.
3. Loading a problem without a schema_version field raises ValueError.
4. Round-trip save/load preserves the schema_version.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qportfolio.problems.portfolio import _SCHEMA_VERSION, PortfolioProblem


@pytest.fixture
def simple_problem():
    rng = np.random.default_rng(0)
    N, K = 4, 2
    tickers = [f"S{i}" for i in range(N)]
    sector_map = {t: "Tech" if i < 2 else "Finance" for i, t in enumerate(tickers)}
    A = rng.normal(0, 0.1, (N, N))
    sigma = (A @ A.T) / N + 0.01 * np.eye(N)
    return PortfolioProblem.from_arrays(
        tickers=tickers,
        mu=rng.uniform(0.1, 0.3, N),
        sigma=sigma,
        sector_map=sector_map,
        K=K,
    )


def test_saved_problem_has_schema_version(simple_problem, tmp_path):
    """problem.json must contain schema_version after save."""
    save_dir = tmp_path / "schema_test"
    simple_problem.save(str(save_dir))

    with open(save_dir / "problem.json") as f:
        meta = json.load(f)

    assert "schema_version" in meta, "problem.json missing schema_version"
    assert meta["schema_version"] == _SCHEMA_VERSION


def test_current_schema_version_is_1_0():
    """The current schema version must be '1.0'."""
    assert _SCHEMA_VERSION == "1.0"


def test_load_unknown_schema_raises(simple_problem, tmp_path):
    """Loading a problem with an unrecognised schema_version raises ValueError."""
    save_dir = tmp_path / "future_schema"
    simple_problem.save(str(save_dir))

    # Tamper with schema_version
    json_path = save_dir / "problem.json"
    with open(json_path) as f:
        meta = json.load(f)
    meta["schema_version"] = "99.0"
    with open(json_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="schema_version"):
        PortfolioProblem.load(str(save_dir))


def test_load_missing_schema_raises(simple_problem, tmp_path):
    """Loading a problem.json with no schema_version field raises ValueError."""
    save_dir = tmp_path / "no_schema"
    simple_problem.save(str(save_dir))

    json_path = save_dir / "problem.json"
    with open(json_path) as f:
        meta = json.load(f)
    del meta["schema_version"]
    with open(json_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="schema_version"):
        PortfolioProblem.load(str(save_dir))


def test_roundtrip_preserves_schema_version(simple_problem, tmp_path):
    """Load → save → load must keep the same schema_version."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    simple_problem.save(str(first_dir))
    loaded = PortfolioProblem.load(str(first_dir))
    loaded.save(str(second_dir))

    with open(second_dir / "problem.json") as f:
        meta = json.load(f)

    assert meta["schema_version"] == _SCHEMA_VERSION


def test_schema_version_in_error_message(simple_problem, tmp_path):
    """ValueError message should name the unsupported version for easy debugging."""
    save_dir = tmp_path / "bad_ver"
    simple_problem.save(str(save_dir))

    json_path = save_dir / "problem.json"
    with open(json_path) as f:
        meta = json.load(f)
    meta["schema_version"] = "0.9"
    with open(json_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="0.9"):
        PortfolioProblem.load(str(save_dir))
