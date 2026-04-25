"""Cache helpers for the weather sub-engine.

Mirrors the on-disk layout of ``engine.cdl`` (single data root, derived/ subdir
for parquet outputs) so all engine sources cache to the same place. Unlike
CDL — which refuses to fall back when EFS isn't mounted because a stray hot
path could pull 9.8 GB — weather caches are small per-county API pulls, so
this module *does* auto-create the data root on first use. That keeps laptops,
CI, and any non-EFS box productive without a mount step.

Layout:
    <data_root>/
    └── derived/
        └── weather/
            ├── power_{geoid}_{start}_{end}.parquet
            ├── smap_{geoid}_{start}_{end}.parquet
            ├── sentinel_{geoid}_{start}_{end}.parquet
            └── merged_{nrows}_{geoid_hash}_{start}_{end}.parquet
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable


def data_root() -> Path:
    """Resolve the engine data root.

    Priority: ``$HACK26_CDL_DATA_DIR`` (shared with the CDL engine so all
    derived parquets colocate next to the rasters on the workshop EFS), then
    ``$HACK26_CACHE_DIR``, then ``~/hack26/data``. Auto-created — weather
    pulls are small per-county API hits, not multi-GB rasters, so silent
    creation is the right default here (CDL is the strict-mode exception).
    """
    env = os.environ.get("HACK26_CDL_DATA_DIR") or os.environ.get("HACK26_CACHE_DIR")
    root = Path(env) if env else Path.home() / "hack26" / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def derived_dir() -> Path:
    """``<data_root>/derived/weather/``. Auto-created on first use."""
    d = data_root() / "derived" / "weather"
    d.mkdir(parents=True, exist_ok=True)
    return d


def power_cache_path(geoid: str, start_year: int, end_year: int) -> Path:
    return derived_dir() / f"power_{geoid}_{start_year}_{end_year}.parquet"


def smap_cache_path(geoid: str, start_year: int, end_year: int) -> Path:
    return derived_dir() / f"smap_{geoid}_{start_year}_{end_year}.parquet"


def sentinel_cache_path(geoid: str, start_date: str, end_date: str) -> Path:
    # Dates are ISO YYYY-MM-DD; strip dashes so the filename is shell-friendly.
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    return derived_dir() / f"sentinel_{geoid}_{s}_{e}.parquet"


def merged_cache_path(
    geoids: Iterable[str], start_year: int, end_year: int, suffix: str = "daily"
) -> Path:
    """Content-addressed cache for a multi-county merged frame.

    Hash is over the *sorted* geoid list so row-order in the input GeoDataFrame
    doesn't cause a false miss, and so two different county sets of the same
    size (e.g. 5 Iowa vs 5 Colorado) don't collide on a single cache file.
    """
    geoids_sorted = sorted(str(g) for g in geoids)
    h = hashlib.sha1("\n".join(geoids_sorted).encode("utf-8")).hexdigest()[:12]
    return derived_dir() / (
        f"weather_{suffix}_{len(geoids_sorted)}_{h}_{start_year}_{end_year}.parquet"
    )
