"""Sentinel-2 NDVI / NDWI per county via Microsoft Planetary Computer STAC.

For each county we hit the ``sentinel-2-l2a`` collection over the county's
geometric bounding box, compute scene-level NDVI and NDWI, and reduce to
one (date, geoid) row per usable scene. Sparse cadence (~5-day revisit,
plus cloud filtering) is the norm — callers typically forward-fill the
result against a daily POWER frame, see :func:`engine.weather.core.merge_weather`.

Determinism contract:
    - Geometry → bbox via ``shapely.geometry.bounds`` (deterministic).
    - Cloud cover threshold and CRS resolution are constants on this module.
    - Cache key is ``(geoid, start_date, end_date)``.

Heavy stack — ``pystac_client`` / ``stackstac`` / ``planetary_computer`` —
is imported lazily so the rest of the weather engine still works on a box
where these aren't installed; the function returns an empty frame and
prints a warning instead of crashing the merge.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Iterable

import pandas as pd

from ._cache import sentinel_cache_path

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_COLLECTION = "sentinel-2-l2a"

# Cloud-cover ceiling in percent; matches the original field-level script.
CLOUD_COVER_LT = 20

# Sentinel-2 was first declared operational in mid-2015; earlier dates just
# return zero scenes, so we silently floor the search start to this date.
SENTINEL_FIRST_DATE = "2015-07-01"

# Bands we need:
#   B04 Red, B08 NIR (NDVI),  B11 SWIR-1 (NDWI/water).
# B8A is included for parity with the original script in case future indices
# (e.g. NDRE) want it; it's a cheap addition since stackstac fetches lazily.
BANDS = ("B04", "B08", "B11", "B8A")


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        {"NDVI": pd.Series(dtype="float64"), "NDWI": pd.Series(dtype="float64")},
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "geoid"]),
    )


def _geometry_bbox(geometry) -> tuple[float, float, float, float]:
    """``(minx, miny, maxx, maxy)`` in the geometry's native CRS (EPSG:4269 for
    counties — close enough to 4326 for STAC bbox search at county scale)."""
    minx, miny, maxx, maxy = geometry.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def fetch_county_sentinel(
    geoid: str,
    geometry,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    cloud_cover_lt: int = CLOUD_COVER_LT,
    resolution_m: int = 30,
) -> pd.DataFrame:
    """NDVI/NDWI for one county over ``[start_date, end_date]``.

    Returns a DataFrame indexed by ``(date, geoid)`` with float NDVI and NDWI.
    Result is one row per Sentinel-2 scene that intersects the county bbox
    and satisfies the cloud-cover filter; the cadence is intentionally sparse
    so the caller can decide how to align it with daily weather (typically a
    per-county forward-fill — see :mod:`engine.weather.core`).

    ``resolution_m`` defaults to 30 m to keep county-scale aggregation tractable
    (a ~50 km county at native 10 m = ~25 M pixels per scene). Bump to 10 if
    you really need full resolution and have the RAM.
    """
    # Floor to Sentinel's earliest operational date so ancient ranges still
    # produce a deterministic empty cache file instead of an opaque API error.
    eff_start = max(start_date, SENTINEL_FIRST_DATE)
    if eff_start > end_date:
        empty = _empty_result()
        return empty

    cache = sentinel_cache_path(geoid, eff_start, end_date)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    try:
        import pystac_client
        import stackstac
    except ImportError as exc:
        print(f"[weather.sentinel] missing dep ({exc.name}); returning empty "
              f"NDVI/NDWI for geoid={geoid}. Install pystac-client + stackstac "
              f"+ planetary-computer to enable.", file=sys.stderr)
        empty = _empty_result()
        empty.to_parquet(cache)
        return empty

    # planetary_computer.sign is required to actually GET the band assets;
    # without it stackstac will hit anonymous URLs and 401 on the rasters.
    try:
        import planetary_computer as pc
        modifier = pc.sign_inplace
    except ImportError:
        modifier = None
        print("[weather.sentinel] planetary-computer not installed; asset URLs "
              "will be unsigned and may 401. `pip install planetary-computer`.",
              file=sys.stderr)

    bbox = _geometry_bbox(geometry)

    catalog = pystac_client.Client.open(STAC_URL, modifier=modifier)
    search = catalog.search(
        collections=[STAC_COLLECTION],
        bbox=bbox,
        datetime=f"{eff_start}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
    )
    items = list(search.items())
    if not items:
        print(f"[weather.sentinel] no scenes for geoid={geoid} "
              f"({eff_start}..{end_date}, cloud<{cloud_cover_lt}%)",
              file=sys.stderr)
        empty = _empty_result()
        empty.to_parquet(cache)
        return empty

    try:
        stack = stackstac.stack(
            items,
            assets=list(BANDS),
            bounds_latlon=bbox,
            resolution=resolution_m,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[weather.sentinel] stackstac.stack failed for geoid={geoid}: {exc}",
              file=sys.stderr)
        empty = _empty_result()
        empty.to_parquet(cache)
        return empty

    red = stack.sel(band="B04").astype("float32")
    nir = stack.sel(band="B08").astype("float32")
    swir = stack.sel(band="B11").astype("float32")

    # +1e-10 guards against /0 over masked-out pixels.
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndwi = (nir - swir) / (nir + swir + 1e-10)

    rows: list[dict] = []
    for scene in stack.time.values:
        rows.append({
            "date": pd.to_datetime(str(scene)[:10]),
            "geoid": str(geoid),
            "NDVI": float(ndvi.sel(time=scene).mean()),
            "NDWI": float(ndwi.sel(time=scene).mean()),
        })

    df = (
        pd.DataFrame(rows)
        .set_index(["date", "geoid"])
        .sort_index()
    )
    df.to_parquet(cache)
    return df


def fetch_counties_sentinel(
    counties,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    cloud_cover_lt: int = CLOUD_COVER_LT,
    resolution_m: int = 30,
    progress_every: int = 5,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Vectorized Sentinel-2 over a county GeoDataFrame.

    Each county is independent and individually cached; failures degrade to
    an empty per-county frame so one bad polygon doesn't poison the batch.
    When ``max_workers`` > 1, counties are fetched concurrently.
    """
    rows = list(counties.iterrows())
    n = len(rows)
    if n == 0:
        return _empty_result()

    def _one(entry: tuple[int, tuple]) -> pd.DataFrame | None:
        _, (_, row) = entry
        return fetch_county_sentinel(
            geoid=str(row["geoid"]),
            geometry=row.geometry,
            start_date=start_date,
            end_date=end_date,
            refresh=refresh,
            cloud_cover_lt=cloud_cover_lt,
            resolution_m=resolution_m,
        )

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            dlist = list(ex.map(_one, list(enumerate(rows))))
        frames = [df for df in dlist if df is not None and not df.empty]
        print(
            f"[weather.sentinel] {len(frames)}/{n} counties with data "
            f"(max_workers={max_workers})",
            file=sys.stderr,
        )
        if not frames:
            return _empty_result()
        return pd.concat(frames).sort_index()

    frames = []
    for i, (_, row) in enumerate(rows, start=1):
        df = fetch_county_sentinel(
            geoid=str(row["geoid"]),
            geometry=row.geometry,
            start_date=start_date,
            end_date=end_date,
            refresh=refresh,
            cloud_cover_lt=cloud_cover_lt,
            resolution_m=resolution_m,
        )
        if not df.empty:
            frames.append(df)
        if i % progress_every == 0 or i == n:
            print(f"[weather.sentinel] {i}/{n} counties processed",
                  file=sys.stderr)

    if not frames:
        return _empty_result()
    return pd.concat(frames).sort_index()
