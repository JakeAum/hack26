"""County Catalog — single source of truth for ROIs.

Loads the Census Bureau TIGER/Line 2024 county shapefile, filters to the five
target states, normalizes columns, and persists a GeoParquet lookup table so
subsequent calls are an in-memory file read.

Public surface:
    load_counties(states=None, refresh=False) -> geopandas.GeoDataFrame

CLI:
    python -m engine.counties              # download + cache, print summary
    python -m engine.counties --refresh    # force re-download
    python -m engine.counties --out ...    # also write a CSV/Parquet copy
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import requests

TIGER_YEAR = 2024
TIGER_URL = (
    f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}"
    f"/COUNTY/tl_{TIGER_YEAR}_us_county.zip"
)

# State FIPS -> human-readable name. This is the authoritative scope for the project.
TARGET_STATES: dict[str, str] = {
    "08": "Colorado",
    "19": "Iowa",
    "29": "Missouri",
    "31": "Nebraska",
    "55": "Wisconsin",
}

_NAME_TO_FIPS = {v.lower(): k for k, v in TARGET_STATES.items()}


def _cache_dir() -> Path:
    # Default to ~/hack26/data so the SageMaker workshop box keeps all engine
    # caches on the mounted EFS volume alongside the pre-staged CDL raster.
    # Override with HACK26_CACHE_DIR for laptops / CI / anywhere without EFS.
    root = Path(os.environ.get("HACK26_CACHE_DIR", Path.home() / "hack26" / "data"))
    d = root / "tiger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _zip_path() -> Path:
    return _cache_dir() / f"tl_{TIGER_YEAR}_us_county.zip"


def _parquet_path() -> Path:
    return _cache_dir() / f"counties_5state_{TIGER_YEAR}.parquet"


def _download_tiger(force: bool = False) -> Path:
    """Fetch the national county shapefile zip into the cache. Idempotent."""
    target = _zip_path()
    if target.exists() and not force:
        return target

    print(f"[counties] downloading {TIGER_URL}", file=sys.stderr)
    with requests.get(TIGER_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = target.with_suffix(".zip.partial")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        tmp.replace(target)

    # Sanity-check that it's a real zip before we trust the cache.
    with zipfile.ZipFile(target) as zf:
        if not any(n.endswith(".shp") for n in zf.namelist()):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded file is not a TIGER shapefile zip: {TIGER_URL}")

    return target


def _resolve_states(states: Iterable[str] | None) -> list[str]:
    """Accept state names or 2-digit FIPS; return list of FIPS codes.

    `None` means "all target states" (the documented default). An *empty*
    iterable is treated as a user error and raises, so callers like
    ``--states`` with no arguments fail loudly instead of silently returning
    zero counties.
    """
    if states is None:
        return list(TARGET_STATES.keys())

    # Materialize once so we can both check emptiness and iterate.
    requested = [str(s).strip() for s in states]
    if not requested:
        raise ValueError(
            "states must be None (for all 5 target states) or a non-empty "
            "iterable of state names / 2-digit FIPS; got an empty list"
        )

    out: list[str] = []
    for s in requested:
        if s in TARGET_STATES:
            out.append(s)
            continue
        fips = _NAME_TO_FIPS.get(s.lower())
        if fips is None:
            raise ValueError(
                f"unknown state {s!r}; expected one of "
                f"{sorted(TARGET_STATES.values())} or FIPS in {sorted(TARGET_STATES)}"
            )
        out.append(fips)
    # Preserve input order, drop dupes.
    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]


def _build_lookup(force_download: bool = False) -> gpd.GeoDataFrame:
    """Parse the cached TIGER shapefile and normalize to the project schema."""
    zip_path = _download_tiger(force=force_download)

    # geopandas can read a shapefile straight out of a zip via fiona/pyogrio.
    # `as_posix()` keeps the URI valid on Windows, where str(Path) uses backslashes.
    raw = gpd.read_file(f"zip://{zip_path.as_posix()}")

    # Filter to our 5 states *before* doing any other work.
    raw = raw[raw["STATEFP"].isin(TARGET_STATES)].copy()

    gdf = gpd.GeoDataFrame(
        {
            "geoid": raw["GEOID"].astype(str).str.zfill(5),
            "state_fips": raw["STATEFP"].astype(str).str.zfill(2),
            "county_fips": raw["COUNTYFP"].astype(str).str.zfill(3),
            "name": raw["NAME"].astype(str),
            "name_full": raw["NAMELSAD"].astype(str),
            "state_name": raw["STATEFP"].map(TARGET_STATES),
            "centroid_lat": pd.to_numeric(raw["INTPTLAT"], errors="coerce"),
            "centroid_lon": pd.to_numeric(raw["INTPTLON"], errors="coerce"),
            "land_area_m2": pd.to_numeric(raw["ALAND"], errors="coerce").astype("Int64"),
            "water_area_m2": pd.to_numeric(raw["AWATER"], errors="coerce").astype("Int64"),
            "geometry": raw.geometry.values,
        },
        crs=raw.crs,
    )

    gdf = gdf.sort_values(["state_fips", "county_fips"]).reset_index(drop=True)

    # Invariants we rely on downstream.
    assert gdf["geoid"].is_unique, "geoid must be unique across the catalog"
    assert gdf["geoid"].str.len().eq(5).all(), "geoid must be 5 chars (state+county FIPS)"
    assert gdf["geometry"].notna().all(), "every county must have a geometry"

    return gdf


def load_counties(
    states: Iterable[str] | None = None,
    refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Return the canonical county lookup table.

    Args:
        states: iterable of state names ("Iowa") or 2-digit FIPS ("19").
            None returns all five target states.
        refresh: if True, re-download TIGER and rebuild the parquet cache.

    Returns:
        GeoDataFrame keyed by `geoid` (5-digit county FIPS), in TIGER's native
        EPSG:4269 (NAD83). See SPEC §4 for the column contract.
    """
    parquet = _parquet_path()
    if refresh or not parquet.exists():
        # SPEC §4: refresh=True rebuilds *both* cache layers (zip and parquet).
        gdf = _build_lookup(force_download=refresh)
        # GeoParquet so geometry round-trips losslessly.
        gdf.to_parquet(parquet, index=False)
    else:
        gdf = gpd.read_parquet(parquet)

    fips = _resolve_states(states)
    if set(fips) != set(TARGET_STATES):
        gdf = gdf[gdf["state_fips"].isin(fips)].reset_index(drop=True)

    return gdf


def _crs_label(crs) -> str:
    """Compact CRS label; pyproj's str(crs) is a giant PROJJSON dump."""
    if crs is None:
        return "<none>"
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    name = getattr(crs, "name", "?")
    return f"EPSG:{epsg} ({name})" if epsg else name


def _summarize(gdf: gpd.GeoDataFrame) -> str:
    by_state = (
        gdf.groupby(["state_fips", "state_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_counties"})
        .sort_values("state_fips")
    )
    lines = [
        f"counties total: {len(gdf)}",
        f"crs:            {_crs_label(gdf.crs)}",
        f"cache:          {_parquet_path()}",
        "",
        by_state.to_string(index=False),
    ]
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/refresh the county lookup table.")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download TIGER and rebuild the parquet cache.",
    )
    parser.add_argument(
        "--states", nargs="+", default=None, metavar="STATE",
        help="One or more state names or 2-digit FIPS to subset to. "
             "Omit the flag entirely for all 5 target states.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional output path (.parquet or .csv) to also write a copy to.",
    )
    args = parser.parse_args(argv)

    gdf = load_counties(states=args.states, refresh=args.refresh)
    print(_summarize(gdf))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.out.suffix.lower()
        if suffix == ".parquet":
            gdf.to_parquet(args.out, index=False)
        elif suffix == ".csv":
            # Drop geometry for CSV; keep centroids + ids.
            gdf.drop(columns="geometry").to_csv(args.out, index=False)
        else:
            raise SystemExit(f"unsupported --out suffix: {suffix} (use .parquet or .csv)")
        print(f"\nwrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
