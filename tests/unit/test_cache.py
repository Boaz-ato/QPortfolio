"""Unit tests for DiskCache."""

from __future__ import annotations

import pandas as pd

from qportfolio.data.cache import DiskCache


def test_cache_miss_returns_none(tmp_cache_dir):
    result = tmp_cache_dir.get("AAPL", "2020-01-01", "2021-01-01")
    assert result is None


def test_cache_put_then_get_roundtrip(tmp_cache_dir, tiny_prices_df):
    """Parquet round-trip preserves DatetimeIndex and float64 values exactly."""
    df = tiny_prices_df[["AAPL"]]
    tmp_cache_dir.put("AAPL", "2022-01-01", "2022-12-31", df)
    recovered = tmp_cache_dir.get("AAPL", "2022-01-01", "2022-12-31")
    assert recovered is not None
    # check_freq=False because Parquet doesn't preserve BusinessDay freq on DatetimeIndex
    pd.testing.assert_frame_equal(df, recovered, check_freq=False)


def test_cache_key_is_deterministic(tmp_path):
    c1 = DiskCache(str(tmp_path / "c1"))
    c2 = DiskCache(str(tmp_path / "c2"))
    k1 = c1._key("AAPL", "2020-01-01", "2021-01-01", "prices")
    k2 = c2._key("AAPL", "2020-01-01", "2021-01-01", "prices")
    assert k1 == k2


def test_cache_different_tickers_dont_collide(tmp_cache_dir):
    df_aapl = pd.DataFrame({"AAPL": [100.0, 101.0]})
    df_msft = pd.DataFrame({"MSFT": [200.0, 202.0]})
    tmp_cache_dir.put("AAPL", "2022-01-01", "2022-12-31", df_aapl)
    tmp_cache_dir.put("MSFT", "2022-01-01", "2022-12-31", df_msft)
    r_aapl = tmp_cache_dir.get("AAPL", "2022-01-01", "2022-12-31")
    r_msft = tmp_cache_dir.get("MSFT", "2022-01-01", "2022-12-31")
    assert "AAPL" in r_aapl.columns
    assert "MSFT" in r_msft.columns


def test_cache_invalidate_removes_file(tmp_cache_dir, tiny_prices_df):
    df = tiny_prices_df[["AAPL"]]
    tmp_cache_dir.put("AAPL", "2022-01-01", "2022-12-31", df)
    existed = tmp_cache_dir.invalidate("AAPL", "2022-01-01", "2022-12-31")
    assert existed is True
    # Now it should be a miss
    assert tmp_cache_dir.get("AAPL", "2022-01-01", "2022-12-31") is None


def test_cache_invalidate_nonexistent_returns_false(tmp_cache_dir):
    result = tmp_cache_dir.invalidate("GOOG", "2020-01-01", "2021-01-01")
    assert result is False


def test_cache_clear_all(tmp_cache_dir, tiny_prices_df):
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        df = tiny_prices_df[[ticker]] if ticker in tiny_prices_df.columns else tiny_prices_df.iloc[:, :1]
        tmp_cache_dir.put(ticker, "2022-01-01", "2022-12-31", df)
    tmp_cache_dir.clear_all()
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        assert tmp_cache_dir.get(ticker, "2022-01-01", "2022-12-31") is None


def test_cache_list_entries(tmp_cache_dir, tiny_prices_df):
    df = tiny_prices_df[["AAPL"]]
    tmp_cache_dir.put("AAPL", "2022-01-01", "2022-12-31", df)
    entries = tmp_cache_dir.list_entries()
    assert len(entries) == 1
    assert entries[0]["ticker"] == "AAPL"
    assert "downloaded_at" in entries[0]
    assert entries[0]["size_bytes"] > 0


def test_cache_different_data_types_dont_collide(tmp_cache_dir):
    """Same ticker+dates with different data_type should be separate entries."""
    df1 = pd.DataFrame({"price": [100.0]})
    df2 = pd.DataFrame({"sector": ["Technology"]})
    tmp_cache_dir.put("AAPL", "2022-01-01", "2022-12-31", df1, "prices")
    tmp_cache_dir.put("AAPL", "2022-01-01", "2022-12-31", df2, "sector")
    entries = tmp_cache_dir.list_entries()
    assert len(entries) == 2
