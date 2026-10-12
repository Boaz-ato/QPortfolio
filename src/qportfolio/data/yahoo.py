"""Market-data provider wrapping yfinance with retry, validation, and caching."""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from qportfolio.config import DataConfig
from qportfolio.data.cache import DiskCache
from qportfolio.data.preprocessing import validate_price_data
from qportfolio.utils.logging import get_logger
from qportfolio.utils.validation import DataValidationError, validate_date_range

_log = get_logger("data.yahoo")


class MarketDataProvider:
    """Wraps yfinance with retry, validation, and optional caching.

    Parameters
    ----------
    config:
        Data retrieval and validation configuration.
    cache:
        Optional DiskCache for persisting downloaded data.
    """

    def __init__(
        self,
        config: DataConfig | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self._config = config or DataConfig()
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Download adjusted close prices for a list of tickers.

        Returns a DataFrame with DatetimeIndex and tickers as columns.
        Columns are ordered exactly as the input `tickers` list.

        Parameters
        ----------
        tickers:
            List of ticker symbols.
        start, end:
            Date range in YYYY-MM-DD format.

        Returns
        -------
        pd.DataFrame
            Adjusted close prices.

        Raises
        ------
        DataValidationError
            If any ticker fails minimum trading-day or missing-value checks.
        """

        validate_date_range(start, end)

        # Try cache for each ticker individually; fall back to bulk download
        cached_frames: dict[str, pd.DataFrame] = {}
        missing_tickers: list[str] = []

        if self._cache is not None:
            for ticker in tickers:
                df = self._cache.get(ticker, start, end, "prices")
                if df is not None:
                    cached_frames[ticker] = df
                else:
                    missing_tickers.append(ticker)
        else:
            missing_tickers = list(tickers)

        # Download missing tickers
        if missing_tickers:
            _log.info(
                "Downloading prices for %d tickers: %s...", len(missing_tickers), missing_tickers[:5]
            )
            raw = self._download_with_retry(missing_tickers, start, end)
            # Store in cache (skip all-NaN columns — bad download, don't poison cache)
            if self._cache is not None:
                for ticker in missing_tickers:
                    if ticker in raw.columns and not raw[ticker].isna().all():
                        self._cache.put(ticker, start, end, raw[[ticker]], "prices")

            for ticker in missing_tickers:
                if ticker not in raw.columns:
                    raise DataValidationError(
                        f"Ticker '{ticker}' was not returned by yfinance. "
                        "It may be delisted or the symbol is incorrect."
                    )
                cached_frames[ticker] = raw[[ticker]]

        # Assemble in original ticker order
        prices = pd.concat([cached_frames[t] for t in tickers], axis=1)
        prices.columns = tickers

        # Forward-fill small gaps (e.g. holidays creating NaN)
        prices = prices.ffill()

        # Validate
        validate_price_data(
            prices,
            min_trading_days=self._config.min_trading_days,
            max_missing_fraction=self._config.max_missing_fraction,
        )

        _log.info(
            "Fetched prices for %d tickers, %d trading days (%s to %s)",
            len(tickers),
            len(prices),
            prices.index[0].date(),
            prices.index[-1].date(),
        )
        return prices

    def fetch_sector_map(self, tickers: list[str]) -> dict[str, str]:
        """Return {ticker: sector_name} using yfinance Ticker.info.

        Falls back to "Unknown" if info is unavailable (e.g. rate-limiting).
        Results are cached under data_type="sector".

        Parameters
        ----------
        tickers:
            List of ticker symbols.

        Returns
        -------
        dict[str, str]
            Mapping from ticker to GICS sector (or "Unknown").
        """

        sector_map: dict[str, str] = {}
        for ticker in tickers:
            # Try cache
            if self._cache is not None:
                cached = self._cache.get(ticker, "sector", "sector", "sector")
                if cached is not None:
                    sector_map[ticker] = str(cached.iloc[0, 0])
                    continue

            # Download from yfinance
            sector = "Unknown"
            try:
                info = yf.Ticker(ticker).info
                sector = info.get("sector", "Unknown") or "Unknown"
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not fetch sector for %s: %s", ticker, exc)

            sector_map[ticker] = sector

            # Cache the sector as a single-cell DataFrame
            if self._cache is not None:
                df_sector = pd.DataFrame({"sector": [sector]})
                self._cache.put(ticker, "sector", "sector", df_sector, "sector")

        n_unknown = sum(1 for s in sector_map.values() if s == "Unknown")
        if n_unknown:
            _log.warning("%d/%d tickers have unknown sector", n_unknown, len(tickers))

        return sector_map

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _download_with_retry(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Download prices from yfinance with exponential-backoff retry."""

        cfg = self._config
        last_exc: Exception | None = None

        for attempt in range(cfg.retry_attempts):
            try:
                raw = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )

                # yfinance returns multi-level columns when multiple tickers
                if isinstance(raw.columns, pd.MultiIndex):
                    # Keep only the "Close" level
                    if "Close" in raw.columns.get_level_values(0):
                        raw = raw["Close"]
                    else:
                        raw = raw.droplevel(0, axis=1)

                # Single ticker comes back as a flat DataFrame with ticker name
                if len(tickers) == 1 and tickers[0] not in raw.columns:
                    raw.columns = tickers

                return raw

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < cfg.retry_attempts - 1:
                    wait = cfg.retry_backoff_seconds * (2**attempt)
                    _log.warning(
                        "Download attempt %d/%d failed (%s). Retrying in %.1fs...",
                        attempt + 1,
                        cfg.retry_attempts,
                        exc,
                        wait,
                    )
                    time.sleep(wait)

        raise DataValidationError(
            f"Failed to download prices after {cfg.retry_attempts} attempts. Last error: {last_exc}"
        ) from last_exc
