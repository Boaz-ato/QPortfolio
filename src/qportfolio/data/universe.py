"""Stock-universe selection with deterministic seeding.

Supports three modes:
  - sector_balanced: select N stocks distributed evenly across sectors (default)
  - random: uniform random draw from the eligible pool
  - custom: caller provides an explicit ticker list
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from qportfolio.config import UniverseConfig
from qportfolio.utils.logging import get_logger
from qportfolio.utils.validation import UniverseSelectionError

_log = get_logger("data.universe")

_DEFAULT_REGISTRY = pathlib.Path(__file__).parent / "universe_registry.json"


class StockUniverse:
    """Maintains a versioned pool of eligible tickers.

    Parameters
    ----------
    config:
        Universe selection configuration.
    registry_path:
        Path to a universe_registry.json. Defaults to the bundled registry.
    """

    def __init__(
        self,
        config: UniverseConfig | None = None,
        registry_path: str | pathlib.Path | None = None,
    ) -> None:
        self._config = config or UniverseConfig()
        reg_path = pathlib.Path(registry_path) if registry_path else _DEFAULT_REGISTRY
        with open(reg_path) as f:
            self._registry: dict = json.load(f)
        _log.debug(
            "Loaded universe registry version=%s with %d sectors",
            self._registry.get("version"),
            len(self._registry["sectors"]),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self) -> list[str]:
        """Run the selection algorithm and return a deterministic ticker list.

        Returns
        -------
        list[str]
            Exactly config.n_stocks ticker symbols.

        Raises
        ------
        UniverseSelectionError
            If n_stocks exceeds the available pool or custom list is too short.
        """
        cfg = self._config
        rng = np.random.default_rng(cfg.seed)

        if cfg.mode == "custom":
            return self._custom()
        elif cfg.mode == "random":
            return self._random(rng)
        else:  # sector_balanced
            return self._sector_balanced(rng)

    def available_sectors(self) -> list[str]:
        """Return sorted list of sector names in the registry."""
        return sorted(self._registry["sectors"].keys())

    def available_tickers(self, sector: str | None = None) -> list[str]:
        """Return all eligible tickers, optionally filtered by sector."""
        if sector is not None:
            return list(self._registry["sectors"].get(sector, []))
        return [t for tickers in self._registry["sectors"].values() for t in tickers]

    def registry_version(self) -> str:
        """Return the registry version string."""
        return self._registry.get("version", "unknown")

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _sector_balanced(self, rng: np.random.Generator) -> list[str]:
        """Distribute n_stocks as evenly as possible across sectors."""
        cfg = self._config
        sectors = self.available_sectors()
        n_sectors = len(sectors)
        n_stocks = cfg.n_stocks

        # Base count and remainder
        base, remainder = divmod(n_stocks, n_sectors)

        pool_sizes = {s: len(self._registry["sectors"][s]) for s in sectors}
        for s in sectors[:remainder]:
            needed = base + 1
            if pool_sizes[s] < needed:
                raise UniverseSelectionError(
                    f"Sector '{s}' has only {pool_sizes[s]} tickers, "
                    f"but {needed} are needed for sector-balanced selection of {n_stocks} stocks."
                )
        for s in sectors[remainder:]:
            needed = base
            if pool_sizes[s] < needed:
                raise UniverseSelectionError(
                    f"Sector '{s}' has only {pool_sizes[s]} tickers, "
                    f"but {needed} are needed for sector-balanced selection of {n_stocks} stocks."
                )

        selected: list[str] = []
        for i, s in enumerate(sectors):
            n = base + (1 if i < remainder else 0)
            if n == 0:
                continue
            pool = self._registry["sectors"][s]
            idx = rng.choice(len(pool), size=n, replace=False)
            selected.extend(pool[j] for j in sorted(idx))

        _log.debug("sector_balanced selected %d stocks across %d sectors", len(selected), n_sectors)
        return selected

    def _random(self, rng: np.random.Generator) -> list[str]:
        """Uniform random draw from the flattened eligible pool."""
        all_tickers = self.available_tickers()
        n = self._config.n_stocks
        if n > len(all_tickers):
            raise UniverseSelectionError(
                f"Requested {n} stocks but only {len(all_tickers)} are in the registry."
            )
        idx = rng.choice(len(all_tickers), size=n, replace=False)
        selected = [all_tickers[i] for i in sorted(idx)]
        _log.debug("random selected %d stocks", len(selected))
        return selected

    def _custom(self) -> list[str]:
        """Return the user-provided ticker list after validation."""
        tickers = self._config.custom_tickers
        if not tickers:
            raise UniverseSelectionError("mode='custom' requires a non-empty custom_tickers list.")
        return list(tickers)
