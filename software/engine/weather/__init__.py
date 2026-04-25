"""Weather sub-engine — NASA POWER + SMAP + Sentinel-2, keyed by county geoid.

Implements the SPEC §2 contract for weather data:

    fetch(geoid, geometry, date_range) -> pd.DataFrame

Lookups are deterministic — for a given county and date range, the same
representative ``(lat, lon)`` (TIGER interior point or polygon centroid),
the same NASA POWER grid cell, and the same parquet cache file are used,
so two calls return byte-identical frames once the cache is warm.

Public surface (lazy-imported to keep ``import engine.weather`` cheap when
only inspecting URLs/constants from another file):

    Per-county fetchers
    -------------------
    fetch_county_power     (geoid, geometry, start_year, end_year, ...)
    fetch_county_smap      (geoid, geometry, start_year, end_year, ...)
    fetch_county_sentinel  (geoid, geometry, start_date, end_date, ...)
    fetch_county_weather   (geoid, geometry, start_year, end_year, ...)

    Vectorized over a county GeoDataFrame
    -------------------------------------
    fetch_counties_power
    fetch_counties_smap
    fetch_counties_sentinel
    fetch_counties_weather

    Feature engineering on a merged daily frame
    -------------------------------------------
    compute_gdd
    add_rolling_features
    build_annual_summary

    Misc
    ----
    merge_weather  — combine power + smap + sentinel on (date, geoid)

CLI:
    python -m engine.weather --states Iowa --start 2020 --end 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    # per-county
    "fetch_county_power",
    "fetch_county_smap",
    "fetch_county_sentinel",
    "fetch_county_weather",
    # vectorized
    "fetch_counties_power",
    "fetch_counties_smap",
    "fetch_counties_sentinel",
    "fetch_counties_weather",
    # features
    "compute_gdd",
    "add_rolling_features",
    "build_annual_summary",
    # merge
    "merge_weather",
    # constants
    "ALL_NASA_PARAMS",
    "NASA_MOISTURE_PARAMS",
    "NASA_SOIL_PARAMS",
    "NASA_TEMP_PARAMS",
    "SMAP_FIRST_YEAR",
    "SENTINEL_FIRST_DATE",
]

# Mirror of the CDL package's lazy-export trick: keep `import engine.weather`
# fast and avoid pulling pystac_client / stackstac into memory unless the
# caller actually needs Sentinel-2.
_LAZY: dict[str, tuple[str, str]] = {
    "fetch_county_power":      ("engine.weather.power",    "fetch_county_power"),
    "fetch_county_smap":       ("engine.weather.power",    "fetch_county_smap"),
    "fetch_county_sentinel":   ("engine.weather.sentinel", "fetch_county_sentinel"),
    "fetch_county_weather":    ("engine.weather.core",     "fetch_county_weather"),
    "fetch_counties_power":    ("engine.weather.power",    "fetch_counties_power"),
    "fetch_counties_smap":     ("engine.weather.power",    "fetch_counties_smap"),
    "fetch_counties_sentinel": ("engine.weather.sentinel", "fetch_counties_sentinel"),
    "fetch_counties_weather":  ("engine.weather.core",     "fetch_counties_weather"),
    "compute_gdd":             ("engine.weather.features", "compute_gdd"),
    "add_rolling_features":    ("engine.weather.features", "add_rolling_features"),
    "build_annual_summary":    ("engine.weather.features", "build_annual_summary"),
    "merge_weather":           ("engine.weather.core",     "merge_weather"),
    "ALL_NASA_PARAMS":         ("engine.weather.power",    "ALL_NASA_PARAMS"),
    "NASA_MOISTURE_PARAMS":    ("engine.weather.power",    "NASA_MOISTURE_PARAMS"),
    "NASA_SOIL_PARAMS":        ("engine.weather.power",    "NASA_SOIL_PARAMS"),
    "NASA_TEMP_PARAMS":        ("engine.weather.power",    "NASA_TEMP_PARAMS"),
    "SMAP_FIRST_YEAR":         ("engine.weather.power",    "SMAP_FIRST_YEAR"),
    "SENTINEL_FIRST_DATE":     ("engine.weather.sentinel", "SENTINEL_FIRST_DATE"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'engine.weather' has no attribute {name!r}")
    mod_name, attr = target
    import importlib
    return getattr(importlib.import_module(mod_name), attr)


if TYPE_CHECKING:
    from engine.weather.core import (
        fetch_counties_weather,
        fetch_county_weather,
        merge_weather,
    )
    from engine.weather.features import (
        add_rolling_features,
        build_annual_summary,
        compute_gdd,
    )
    from engine.weather.power import (
        ALL_NASA_PARAMS,
        NASA_MOISTURE_PARAMS,
        NASA_SOIL_PARAMS,
        NASA_TEMP_PARAMS,
        SMAP_FIRST_YEAR,
        fetch_counties_power,
        fetch_counties_smap,
        fetch_county_power,
        fetch_county_smap,
    )
    from engine.weather.sentinel import (
        SENTINEL_FIRST_DATE,
        fetch_counties_sentinel,
        fetch_county_sentinel,
    )
