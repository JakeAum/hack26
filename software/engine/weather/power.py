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

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import pandas as pd
import requests

from ._cache import power_cache_path, smap_cache_path

# Engine-wide logger so SMAP/POWER warnings land in the same rotating log file
# as the dataset/model/forecast pipeline. Falls back to a stderr-only logger
# if engine._logging hasn't been initialized yet.
try:
    from .._logging import get_logger as _get_logger  # type: ignore
    _LOG = _get_logger(__name__)
except Exception:  # noqa: BLE001 — engine._logging is optional at import time
    _LOG = logging.getLogger(__name__)

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
# SMAP / SMLAND degradation flag (process-local).
#
# NASA POWER periodically deprecates / renames SMAP-derived parameters, and
# during such transitions every ``SMLAND`` request 422s. We don't want a cold
# pull to spend ~99 × 1 s of useless rate-limited HTTP calls + flood stderr,
# and we don't want users to think the run is broken (the dataset code already
# tolerates SMAP being missing — the column is dropped from past covariates).
#
# After the first 422 we set ``_SMAP_BROKEN_THIS_PROCESS`` and short-circuit
# every subsequent SMAP call: write an empty cache parquet and return empty.
# A single banner-level log line tells the user what happened, with the URL
# of the failing probe so they can rerun it manually if they want to confirm
# POWER is back.
# ---------------------------------------------------------------------------
_SMAP_BROKEN_THIS_PROCESS: bool = False
_SMAP_BREAK_REASON: str = ""
# SMAP is probed from fetch_counties_smap with ThreadPoolExecutor; without a
# lock, N threads can hit NASA before the first 422 sets _SMAP_BROKEN_*,
# causing duplicate 422 spam. Serialize the failure path and post-failure
# short-circuits.
_SMAP_LOCK = threading.Lock()


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


def _empty_smap_frame(cache_path=None) -> pd.DataFrame:
    """Return (and optionally cache) an empty SMAP frame with the right index."""
    empty = pd.DataFrame(index=pd.MultiIndex.from_arrays(
        [[], []], names=["date", "geoid"]
    ))
    if cache_path is not None:
        try:
            empty.to_parquet(cache_path)
        except Exception:  # noqa: BLE001 — cache write is best-effort
            pass
    return empty


def _mark_smap_broken(reason: str) -> None:
    """Trip the process-local SMAP-broken flag once, with a banner log."""
    global _SMAP_BROKEN_THIS_PROCESS, _SMAP_BREAK_REASON
    if _SMAP_BROKEN_THIS_PROCESS:
        return
    _SMAP_BROKEN_THIS_PROCESS = True
    _SMAP_BREAK_REASON = reason
    _LOG.warning(
        "NASA POWER SMAP/SMLAND is rejecting requests this run; "
        "skipping all subsequent SMAP fetches in this process. "
        "Reason: %s. The model trains fine without SMAP — the "
        "SMAP_surface_sm_m3m3 past-covariate column will be empty and "
        "automatically dropped downstream. Pass --no-smap to skip the probe "
        "entirely on the next run.",
        reason,
    )


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

    If the NASA POWER ``SMLAND`` parameter is rejected once during this
    process (HTTP 422 or similar), every subsequent call short-circuits to an
    empty frame so we don't spend the rest of the cold pull on dead requests.
    Re-run after restarting Python to retry SMAP from scratch.
    """
    effective_start = max(start_year, SMAP_FIRST_YEAR)
    if effective_start > end_year:
        return _empty_smap_frame()

    cache = smap_cache_path(geoid, effective_start, end_year)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    if _SMAP_BROKEN_THIS_PROCESS:
        return _empty_smap_frame(cache)

    with _SMAP_LOCK:
        if _SMAP_BROKEN_THIS_PROCESS:
            return _empty_smap_frame(cache)
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)

        lat, lon = representative_latlon(geometry, county_row=county_row)
        try:
            raw = _fetch_power_point(
                lat, lon, ("SMLAND",), effective_start, end_year,
            )
        except requests.HTTPError as exc:  # 422 / 4xx → degrade for the whole run
            status = getattr(exc.response, "status_code", "?")
            body = ""
            if exc.response is not None:
                try:
                    body = exc.response.text[:200].replace("\n", " ")
                except Exception:  # noqa: BLE001
                    body = ""
            reason = (
                f"HTTP {status} for SMLAND@({lat:.3f},{lon:.3f}) "
                f"{effective_start}-{end_year}; body={body!r}"
            )
            # One banner from _mark_smap_broken — no per-geoid line (parallel
            # SMAP would duplicate this N times before the flag existed).
            _mark_smap_broken(reason)
            return _empty_smap_frame(cache)
        except Exception as exc:  # noqa: BLE001 — SMAP gaps are common; degrade.
            _LOG.warning("[weather.power] SMAP fetch failed for geoid=%s "
                         "(%.3f,%.3f): %s", geoid, lat, lon, exc)
            return _empty_smap_frame(cache)
        else:
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
    max_workers: int = 1,
) -> pd.DataFrame:
    """Loop :func:`fetch_county_power` over every row in a county GeoDataFrame.

    ``sleep_between`` keeps us polite to NASA POWER (no documented rate limit
    but the original script used 1 s; cached counties skip the sleep entirely).
    When ``max_workers`` > 1, counties are fetched concurrently (I/O bound);
    ``sleep_between`` is not applied in that mode — concurrency caps load on NASA.
    """
    rows = list(counties.iterrows())
    n = len(rows)
    if n == 0:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=["date", "geoid"]
        ))

    if max_workers > 1:
        def _one(entry: tuple[int, tuple]) -> tuple[int, pd.DataFrame | None]:
            j, (_, row) = entry
            geoid = str(row["geoid"])
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
                return (j, df)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "[weather.power] POWER failed for geoid=%s: %s",
                    geoid, exc,
                )
                return (j, None)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # map preserves input order
            out = list(ex.map(_one, list(enumerate(rows))))
        frames = [f for _, f in out if f is not None]
        _LOG.info(
            "[weather.power] %d/%d counties processed (max_workers=%d)",
            len(frames), n, max_workers,
        )
        if not frames:
            return pd.DataFrame(index=pd.MultiIndex.from_arrays(
                [[], []], names=["date", "geoid"]
            ))
        return pd.concat(frames).sort_index()

    frames = []
    for i, (_, row) in enumerate(rows, start=1):
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
            _LOG.warning("[weather.power] POWER failed for geoid=%s: %s",
                         geoid, exc)
            continue
        frames.append(df)
        if i % progress_every == 0 or i == n:
            _LOG.info("[weather.power] %d/%d counties processed", i, n)
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
    max_workers: int = 1,
) -> pd.DataFrame:
    """Vectorized SMAP — same pattern as :func:`fetch_counties_power`.

    Once the per-process SMAP-broken flag is tripped (see
    :func:`fetch_county_smap`), every subsequent county is short-circuited
    so the cold pull doesn't burn ~1 s per county on dead HTTP calls. The
    return value is still a properly-typed (possibly empty) DataFrame so
    downstream merges don't blow up.
    """
    rows = list(counties.iterrows())
    n = len(rows)
    if n == 0:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=["date", "geoid"]
        ))

    if max_workers > 1:
        def _one(entry: tuple[int, tuple]) -> pd.DataFrame | None:
            _, (_, row) = entry
            geoid = str(row["geoid"])
            return fetch_county_smap(
                geoid=geoid,
                geometry=row.geometry,
                start_year=start_year,
                end_year=end_year,
                county_row=row,
                refresh=refresh,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            dlist = list(ex.map(_one, list(enumerate(rows))))
        frames = [df for df in dlist if not df.empty]
        if _SMAP_BROKEN_THIS_PROCESS:
            _LOG.warning(
                "SMAP unavailable this run (POWER rejected SMLAND); "
                "continuing without SMAP — model still trains. "
                "(parallel fetch, max_workers=%d)",
                max_workers,
            )
        else:
            _LOG.info(
                "[weather.power] SMAP %d/%d counties with data (max_workers=%d)",
                len(frames), n, max_workers,
            )
        if not frames:
            return pd.DataFrame(index=pd.MultiIndex.from_arrays(
                [[], []], names=["date", "geoid"]
            ))
        return pd.concat(frames).sort_index()

    frames = []
    n_short_circuited = 0
    for i, (_, row) in enumerate(rows, start=1):
        geoid = str(row["geoid"])
        cache = smap_cache_path(geoid, max(start_year, SMAP_FIRST_YEAR), end_year)
        had_cache = cache.exists() and not refresh
        was_broken_before = _SMAP_BROKEN_THIS_PROCESS
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
        if was_broken_before:
            n_short_circuited += 1
        if i % progress_every == 0 or i == n:
            tail = (
                f"  ({n_short_circuited} short-circuited after first 422)"
                if _SMAP_BROKEN_THIS_PROCESS else ""
            )
            _LOG.info("[weather.power] SMAP %d/%d counties processed%s",
                      i, n, tail)
        # Don't sleep if SMAP is already broken — short-circuit is instant
        # and the rate-limit was for the network call we're skipping.
        if (not had_cache and not _SMAP_BROKEN_THIS_PROCESS
                and sleep_between and i < n):
            time.sleep(sleep_between)

    if _SMAP_BROKEN_THIS_PROCESS:
        _LOG.warning(
            "SMAP unavailable this run (POWER rejected SMLAND); "
            "%d/%d counties returned empty frames. "
            "Continuing without SMAP — model still trains.",
            n, n,
        )
    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=["date", "geoid"]
        ))
    return pd.concat(frames).sort_index()
