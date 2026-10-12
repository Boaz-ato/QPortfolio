"""File-based disk cache for market data.

Each cache entry is stored as:
  <cache_dir>/<sha256_hex>.parquet   — the DataFrame
  <cache_dir>/<sha256_hex>.meta.json — metadata (ticker, dates, shape, timestamps)

Keys are sha256(f"{ticker}|{start}|{end}|{data_type}"), giving collision-free,
human-opaque filenames. Writes are atomic (tmp file → rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import time
from typing import Any

import pandas as pd

from qportfolio.utils.logging import get_logger

_log = get_logger("data.cache")


class DiskCache:
    """File-based cache keyed by (ticker, start, end, data_type)."""

    def __init__(self, cache_dir: str = "~/.qportfolio/cache") -> None:
        self._dir = pathlib.Path(os.path.expanduser(cache_dir))
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, ticker: str, start: str, end: str, data_type: str = "prices") -> str:
        """Return a deterministic hex key for the given parameters."""
        raw = f"{ticker}|{start}|{end}|{data_type}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _parquet_path(self, key: str) -> pathlib.Path:
        return self._dir / f"{key}.parquet"

    def _meta_path(self, key: str) -> pathlib.Path:
        return self._dir / f"{key}.meta.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, ticker: str, start: str, end: str, data_type: str = "prices") -> pd.DataFrame | None:
        """Return cached DataFrame or None on cache miss."""
        key = self._key(ticker, start, end, data_type)
        path = self._parquet_path(key)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            _log.debug("Cache hit: %s %s %s-%s", data_type, ticker, start, end)
            return df
        except Exception as exc:  # noqa: BLE001
            _log.warning("Cache read failed for %s %s: %s", ticker, data_type, exc)
            return None

    def put(
        self, ticker: str, start: str, end: str, data: pd.DataFrame, data_type: str = "prices"
    ) -> None:
        """Serialize DataFrame to Parquet under cache_dir (atomic write)."""
        key = self._key(ticker, start, end, data_type)
        parquet_path = self._parquet_path(key)
        meta_path = self._meta_path(key)

        # Atomic write: write to tmp then rename
        fd, tmp_parquet = tempfile.mkstemp(dir=self._dir, suffix=".parquet.tmp")
        os.close(fd)
        try:
            data.to_parquet(tmp_parquet)
            os.replace(tmp_parquet, parquet_path)
        except Exception:
            if os.path.exists(tmp_parquet):
                os.unlink(tmp_parquet)
            raise

        meta: dict[str, Any] = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "data_type": data_type,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows": len(data),
            "columns": list(data.columns) if hasattr(data, "columns") else [],
        }
        fd2, tmp_meta = tempfile.mkstemp(dir=self._dir, suffix=".meta.json.tmp")
        os.close(fd2)
        try:
            with open(tmp_meta, "w") as f:
                json.dump(meta, f)
            os.replace(tmp_meta, meta_path)
        except Exception:
            if os.path.exists(tmp_meta):
                os.unlink(tmp_meta)
            raise

        _log.debug("Cache put: %s %s %s-%s (%d rows)", data_type, ticker, start, end, len(data))

    def invalidate(self, ticker: str, start: str, end: str, data_type: str = "prices") -> bool:
        """Delete a single cache entry. Returns True if it existed."""
        key = self._key(ticker, start, end, data_type)
        existed = False
        for path in (self._parquet_path(key), self._meta_path(key)):
            if path.exists():
                path.unlink()
                existed = True
        return existed

    def clear_all(self) -> None:
        """Remove all files in the cache directory."""
        for f in self._dir.iterdir():
            if f.is_file():
                f.unlink()
        _log.info("Cache cleared: %s", self._dir)

    def list_entries(self) -> list[dict[str, Any]]:
        """Return list of metadata dicts for all cached entries."""
        entries = []
        for meta_file in self._dir.glob("*.meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                parquet = meta_file.with_suffix("").with_suffix(".parquet")
                meta["size_bytes"] = parquet.stat().st_size if parquet.exists() else 0
                entries.append(meta)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not read meta file %s: %s", meta_file, exc)
        return entries
