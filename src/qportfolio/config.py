"""Configuration dataclasses for qportfolio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    """Configuration for market-data retrieval and validation."""

    start: str = "2020-01-01"
    end: str = "2025-12-31"
    min_trading_days: int = 200
    max_missing_fraction: float = 0.02
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0
    cache_dir: str = "~/.qportfolio/cache"


@dataclass
class CovarianceConfig:
    """Configuration for covariance estimation and optional sparsification."""

    method: Literal["full", "sparse"] = "full"
    # threshold: a float in [0, 1) or "auto"
    threshold: float | str = "auto"
    validation_fraction: float = 0.2
    max_risk_error: float = 0.05
    n_validation_portfolios: int = 200
    diagonal_loading: float | None = None  # None = auto-select if needed


@dataclass
class UniverseConfig:
    """Configuration for stock-universe selection."""

    mode: Literal["sector_balanced", "random", "custom"] = "sector_balanced"
    n_stocks: int = 40
    seed: int | None = 42
    custom_tickers: list[str] = field(default_factory=list)
    registry_version: str = "v1"
