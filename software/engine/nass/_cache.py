"""On-disk cache for ``engine.nass`` Quick Stats responses.

Layout (same data root as ``engine.weather`` — auto-created, small parquets)::

    <data_root>/
    └── derived/
        └── nass/
            ├── corn_county_yields_{state_ansi}_{start}_{end}.parquet
            └── corn_state_forecasts_{state_ansi}_{start}_{end}.parquet
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """Match ``engine.weather`` / ``engine.cdl`` data root selection."""
    env = os.environ.get("HACK26_CDL_DATA_DIR") or os.environ.get("HACK26_CACHE_DIR")
    root = Path(env) if env else Path.home() / "hack26" / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def derived_dir() -> Path:
    d = data_root() / "derived" / "nass"
    d.mkdir(parents=True, exist_ok=True)
    return d


def county_yields_path(state_ansi: str, start_year: int, end_year: int) -> Path:
    return derived_dir() / f"corn_county_yields_{state_ansi}_{start_year}_{end_year}.parquet"


def state_forecasts_path(state_ansi: str, start_year: int, end_year: int) -> Path:
    return derived_dir() / f"corn_state_forecasts_{state_ansi}_{start_year}_{end_year}.parquet"
