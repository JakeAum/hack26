"""NASA POWER + SMAP point-time-series fetcher, keyed by county geoid.

NASA POWER is a *point* API (0.5° gridded reanalysis). To honor the SPEC §2
contract — every source takes the county geometry — we deterministically
collapse each polygon to a single representative lat/lon (TIGER's
``INTPTLAT``/``INTPTLON`` interior point if available, else ``geometry.centroid``).
The same county at the same date range therefore always hits the same POWER
grid cell, and the result is stable across calls.

Output schema (returned by every public fetch in this module):

    MultiIndex: (date: pd.Timestamp, geoid: str)
    columns:    one float per requested NASA POWER parameter

Parameter groups exported as module constants are the same ones the original
field-level script pulled — moisture/precipitation, soil moisture, temperature
— so downstream feature engineering in :mod:`engine.weather.features` works
unchanged.

Cache:
    <data_root>/derived/weather/power_{geoid}_{start}_{end}.parquet
    <data_root>/derived/weather/smap_{geoid}_{start}_{end}.parquet
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Sequence

import pandas as pd
import requests

from ._cache import power_cache_path, smap_cache_path

# ---------------------------------------------------------------------------
# Parameter groups (same scientific selection as the original field script).
# ---------------------------------------------------------------------------

NASA_MOISTURE_PARAMS: tuple[str, ...] = (
    "PRECTOTCORR",  # Precipitation (mm/day)
    "RH2M",         # Relative humidity at 2 m (%)
    "T2MDEW",       # Dewpoint temperature at 2 m (°C)
    "EVPTRNS",      # Evapotranspiration (mm/day)
)

NASA_SOIL_PARAMS: tuple[str, ...] = (
    "GWETROOT",     # Root-zone soil wetness, ~0–100 cm  (0–1)
    "GWETTOP",      # Surface soil wetness, top 5 cm     (0–1)
    "GWETPROF",     # Full-profile soil wetness          (0–1)
)

NASA_TEMP_PARAMS: tuple[str, ...] = (
    "T2M",          # Mean air temperature at 2 m (°C)
    "T2M_MAX",      # Daily max air temperature at 2 m (°C)
    "T2M_MIN",      # Daily min air temperature at 2 m (°C)
    "TS",           # Earth skin / surface temperature (°C)
    "T10M",         # Temperature at 10 m (°C)
    "FROST_DAYS",   # Monthly frost-day count, repeated daily
)

ALL_NASA_PARAMS: tuple[str, ...] = (
    NASA_MOISTURE_PARAMS + NASA_SOIL_PARAMS + NASA_TEMP_PARAMS
)

# SMAP coverage starts when the satellite did. Older years just return blanks,
# so we skip the API call entirely below this cutoff.
SMAP_FIRST_YEAR = 2015

POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# NASA POWER's documented sentinel for "no data".
_NODATA = -999.0


# ---------------------------------------------------------------------------
# Geometry → representative lat/lon
# ---------------------------------------------------------------------------

def representative_latlon(geometry, county_row=None) -> tuple[float, float]:
    """Pick a single, stable (lat, lon) for a county.

    Prefers the TIGER ``INTPTLAT``/``INTPTLON`` interior point already
    materialized on the County Catalog row (guaranteed to be inside the
    polygon — ``geometry.centroid`` can fall outside concave polygons), and
    falls back to the polygon's centroid otherwise. The same input always
    returns the same point, so the POWER grid-cell lookup is deterministic.
    """
    if county_row is not None:
        lat = county_row.get("centroid_lat") if hasattr(county_row, "get") else None
        lon = county_row.get("centroid_lon") if hasattr(county_row, "get") else None
        if lat is not None and lon is not None and pd.notna(lat) and pd.notna(lon):
            return float(lat), float(lon)
    c = geometry.centroid
    return float(c.y), float(c.x)


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------

def _fetch_power_point(
    lat: float,
    lon: float,
    parameters: Sequence[str],
    start_year: int,
    end_year: int,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Raw call to the NASA POWER daily point endpoint."""
    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": f"{lon:.6f}",
        "latitude": f"{lat:.6f}",
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    resp = requests.get(POWER_BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    param_data = payload["properties"]["parameter"]

    df = pd.DataFrame(param_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df.replace(_NODATA, pd.NA, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public per-county fetchers
# ---------------------------------------------------------------------------

def fetch_county_power(
    geoid: str,
    geometry,
    start_year: int,
    end_year: int,
    parameters: Sequence[str] = ALL_NASA_PARAMS,
    county_row=None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily NASA POWER frame for one county.

    Returns a DataFrame indexed by ``(date, geoid)``. The cache key is
    ``(geoid, start_year, end_year)`` — note that it is NOT keyed on the
    parameter list, so callers asking for a *narrower* slice get a fast cache
    hit but won't accidentally back-fill a *wider* slice; pass
    ``refresh=True`` if you need to pull additional parameters.
    """
    cache = power_cache_path(geoid, start_year, end_year)
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        # Trim to the parameters the caller asked for so the contract matches
        # what they'd get on a cold pull. Anything extra in cache is ignored.
        wanted = [p for p in parameters if p in df.columns]
        return df[wanted].copy() if wanted else df.copy()

    lat, lon = representative_latlon(geometry, county_row=county_row)
    df = _fetch_power_point(lat, lon, parameters, start_year, end_year)
    df = df.assign(geoid=str(geoid)).set_index("geoid", append=True)

    df.to_parquet(cache)
    return df


def fetch_county_smap(
    geoid: str,
    geometry,
    start_year: int,
    end_year: int,
    county_row=None,
    refresh: bool = False,
) -> pd.DataFrame:
    """SMAP-derived surface soil moisture (m³/m³) for one county, 2015+.

    Returns an empty DataFrame (with the right index names) for years before
    :data:`SMAP_FIRST_YEAR` or when the API has no data for the requested
    point. Callers should ``join(..., how="left")`` against the POWER frame
    so missing rows just show up as NaN.
    """
    effective_start = max(start_year, SMAP_FIRST_YEAR)
    if effective_start > end_year:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=["date", "geoid"]
        ))

    cache = smap_cache_path(geoid, effective_start, end_year)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    lat, lon = representative_latlon(geometry, county_row=county_row)
    try:
        raw = _fetch_power_point(
            lat, lon, ("SMLAND",), effective_start, end_year,
        )
    except Exception as exc:  # noqa: BLE001 — SMAP gaps are common; degrade.
        print(f"[weather.power] SMAP fetch failed for geoid={geoid} "
              f"({lat:.3f},{lon:.3f}): {exc}", file=sys.stderr)
        empty = pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=["date", "geoid"]
        ))
        empty.to_parquet(cache)
        return empty

    raw = raw.rename(columns={"SMLAND": "SMAP_surface_sm_m3m3"})
    raw = raw.assign(geoid=str(geoid)).set_index("geoid", append=True)
    raw.to_parquet(cache)
    return raw


# ---------------------------------------------------------------------------
# Vectorized helper over a county GeoDataFrame
# ---------------------------------------------------------------------------

def fetch_counties_power(
    counties,
    start_year: int,
    end_year: int,
    parameters: Sequence[str] = ALL_NASA_PARAMS,
    refresh: bool = False,
    sleep_between: float = 1.0,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Loop :func:`fetch_county_power` over every row in a county GeoDataFrame.

    ``sleep_between`` keeps us polite to NASA POWER (no documented rate limit
    but the original script used 1 s; cached counties skip the sleep entirely).
    """
    frames: list[pd.DataFrame] = []
    n = len(counties)
    for i, (_, row) in enumerate(counties.iterrows(), start=1):
        geoid = str(row["geoid"])
        cache = power_cache_path(geoid, start_year, end_year)
        had_cache = cache.exists() and not refresh
        try:
            df = fetch_county_power(
                geoid=geoid,
                geometry=row.geometry,
                start_year=start_year,
                end_year=end_year,
                parameters=parameters,
                county_row=row,
                refresh=refresh,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[weather.power] POWER failed for geoid={geoid}: {exc}",
                  file=sys.stderr)
            continue
        frames.append(df)
        if i % progress_every == 0 or i == n:
            print(f"[weather.power] {i}/{n} counties processed", file=sys.stderr)
        # Only rate-limit when we actually hit the network.
        if not had_cache and sleep_between and i < n:
            time.sleep(sleep_between)

    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=["date", "geoid"]
        ))
    return pd.concat(frames).sort_index()


def fetch_counties_smap(
    counties,
    start_year: int,
    end_year: int,
    refresh: bool = False,
    sleep_between: float = 1.0,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Vectorized SMAP — same pattern as :func:`fetch_counties_power`."""
    frames: list[pd.DataFrame] = []
    n = len(counties)
    for i, (_, row) in enumerate(counties.iterrows(), start=1):
        geoid = str(row["geoid"])
        cache = smap_cache_path(geoid, max(start_year, SMAP_FIRST_YEAR), end_year)
        had_cache = cache.exists() and not refresh
        df = fetch_county_smap(
            geoid=geoid,
            geometry=row.geometry,
            start_year=start_year,
            end_year=end_year,
            county_row=row,
            refresh=refresh,
        )
        if not df.empty:
            frames.append(df)
        if i % progress_every == 0 or i == n:
            print(f"[weather.power] SMAP {i}/{n} counties processed",
                  file=sys.stderr)
        if not had_cache and sleep_between and i < n:
            time.sleep(sleep_between)

    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=["date", "geoid"]
        ))
    return pd.concat(frames).sort_index()
