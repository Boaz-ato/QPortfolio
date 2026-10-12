"""Unit tests for StockUniverse."""

from __future__ import annotations

import pytest

from qportfolio.config import UniverseConfig
from qportfolio.data.universe import StockUniverse
from qportfolio.utils.validation import UniverseSelectionError


@pytest.fixture
def universe_40():
    cfg = UniverseConfig(mode="sector_balanced", n_stocks=40, seed=42)
    return StockUniverse(cfg)


# ---------------------------------------------------------------------------
# sector_balanced
# ---------------------------------------------------------------------------


def test_sector_balanced_exact_count(universe_40):
    tickers = universe_40.select()
    assert len(tickers) == 40


def test_sector_balanced_5_per_sector(universe_40):
    """40 stocks / 8 sectors = exactly 5 per sector."""
    tickers = universe_40.select()
    # Get sector of each selected ticker from the registry
    sector_counts: dict[str, int] = {}
    all_tickers_by_sector = {}
    for s in universe_40.available_sectors():
        for t in universe_40.available_tickers(s):
            all_tickers_by_sector[t] = s
    for t in tickers:
        s = all_tickers_by_sector.get(t, "Unknown")
        sector_counts[s] = sector_counts.get(s, 0) + 1
    assert all(v == 5 for v in sector_counts.values()), f"Unequal sector counts: {sector_counts}"


def test_sector_balanced_uneven():
    """10 stocks, 8 sectors: 2 sectors get 2, the rest get 1. Total=10."""
    cfg = UniverseConfig(mode="sector_balanced", n_stocks=10, seed=0)
    u = StockUniverse(cfg)
    tickers = u.select()
    assert len(tickers) == 10


def test_sector_balanced_deterministic_with_seed():
    """Same seed → same ticker list."""
    cfg = UniverseConfig(mode="sector_balanced", n_stocks=40, seed=99)
    t1 = StockUniverse(cfg).select()
    t2 = StockUniverse(cfg).select()
    assert t1 == t2


def test_sector_balanced_different_seeds_differ():
    cfg1 = UniverseConfig(mode="sector_balanced", n_stocks=40, seed=1)
    cfg2 = UniverseConfig(mode="sector_balanced", n_stocks=40, seed=2)
    t1 = StockUniverse(cfg1).select()
    t2 = StockUniverse(cfg2).select()
    assert t1 != t2


def test_sector_balanced_error_exceeds_pool():
    """Requesting more stocks than available per sector → error."""
    cfg = UniverseConfig(mode="sector_balanced", n_stocks=100, seed=0)  # only 80 in registry
    u = StockUniverse(cfg)
    with pytest.raises(UniverseSelectionError):
        u.select()


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------


def test_random_mode_count():
    cfg = UniverseConfig(mode="random", n_stocks=20, seed=42)
    tickers = StockUniverse(cfg).select()
    assert len(tickers) == 20


def test_random_mode_deterministic_with_seed():
    cfg = UniverseConfig(mode="random", n_stocks=20, seed=42)
    t1 = StockUniverse(cfg).select()
    t2 = StockUniverse(cfg).select()
    assert t1 == t2


def test_random_mode_different_seeds_differ():
    cfg1 = UniverseConfig(mode="random", n_stocks=20, seed=1)
    cfg2 = UniverseConfig(mode="random", n_stocks=20, seed=2)
    t1 = StockUniverse(cfg1).select()
    t2 = StockUniverse(cfg2).select()
    assert t1 != t2


def test_random_mode_error_exceeds_pool():
    cfg = UniverseConfig(mode="random", n_stocks=100, seed=0)  # only 80 in registry
    with pytest.raises(UniverseSelectionError):
        StockUniverse(cfg).select()


# ---------------------------------------------------------------------------
# custom
# ---------------------------------------------------------------------------


def test_custom_mode_returns_provided_tickers():
    tickers = ["AAPL", "MSFT", "GOOG"]
    cfg = UniverseConfig(mode="custom", custom_tickers=tickers)
    result = StockUniverse(cfg).select()
    assert result == tickers


def test_custom_mode_empty_list_raises():
    cfg = UniverseConfig(mode="custom", custom_tickers=[])
    with pytest.raises(UniverseSelectionError):
        StockUniverse(cfg).select()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_available_sectors_count():
    u = StockUniverse()
    assert len(u.available_sectors()) == 8


def test_registry_version():
    u = StockUniverse()
    assert u.registry_version() == "v1"


def test_available_tickers_count():
    u = StockUniverse()
    all_tickers = u.available_tickers()
    assert len(all_tickers) == 80  # 8 sectors × 10 tickers


def test_available_tickers_by_sector():
    u = StockUniverse()
    tech = u.available_tickers("Technology")
    assert "AAPL" in tech
    assert len(tech) == 10
