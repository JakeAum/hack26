"""End-to-end weather frame for one or many counties.

Combines NASA POWER (daily weather + soil + temperature), SMAP (surface soil
moisture, 2015+), and Sentinel-2 (NDVI/NDWI per scene, 2015+) into a single
tidy DataFrame indexed by ``(date, geoid)``, then layers on derived features
(GDD, GDD-cumulative, 7d/30d rolling means).

This is the file that satisfies the SPEC §2 contract for the weather source:

    fetch(geoid, geometry, date_range) -> pd.DataFrame

Public surface:
    fetch_county_weather(geoid, geometry, start_year, end_year, ...) -> DataFrame
    fetch_counties_weather(counties, start_year, end_year, ..., max_workers=4) -> DataFrame
    merge_weather(power_df, smap_df, sentinel_df) -> DataFrame
    _main(argv) -> int   # CLI: python -m engine.weather

CLI examples:
    python -m engine.weather --states Iowa --start 2020 --end 2024
    python -m engine.weather --states Iowa Colorado --start 2015 --end 2024 \
        --no-sentinel --out iowa_co_weather.parquet
    python -m engine.weather --geoid 19169 --start 2018 --end 2024 \
        --annual-out story_county_annual.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ._cache import merged_cache_path
from .features import add_rolling_features, build_annual_summary, compute_gdd
from .power import (
    ALL_NASA_PARAMS,
    SMAP_FIRST_YEAR,
    fetch_counties_power,
    fetch_counties_smap,
    fetch_county_power,
    fetch_county_smap,
)
from .sentinel import (
    SENTINEL_FIRST_DATE,
    fetch_counties_sentinel,
    fetch_county_sentinel,
)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _ffill_sentinel_to_daily(
    sentinel_df: pd.DataFrame, daily_index: pd.MultiIndex
) -> pd.DataFrame:
    """Reindex sparse Sentinel rows onto the dense daily ``(date, geoid)``
    index, forward-filling within each county.

    Sentinel-2 cadence is ~5 days at the equator and worse with cloud
    filtering, so most daily rows would be NaN otherwise. Forward-fill is
    the standard treatment for vegetation indices in agronomic ML — a leaf
    canopy doesn't change overnight.
    """
    if sentinel_df.empty:
        return pd.DataFrame(index=daily_index, columns=["NDVI", "NDWI"], dtype="float64")

    pieces: list[pd.DataFrame] = []
    daily_dates = daily_index.get_level_values("date")
    daily_geoids = daily_index.get_level_values("geoid")

    for geoid, scenes in sentinel_df.groupby(level="geoid"):
        target_dates = daily_dates[daily_geoids == geoid]
        if len(target_dates) == 0:
            continue
        flat = scenes.reset_index(level="geoid", drop=True).sort_index()
        # First reindex onto union, ffill, then trim to the daily target —
        # this lets us project forward from the most recent prior scene
        # even when no scene falls exactly on the daily target.
        union = flat.index.union(target_dates).unique().sort_values()
        reindexed = flat.reindex(union).ffill().reindex(target_dates)
        reindexed = reindexed.assign(geoid=geoid).set_index("geoid", append=True)
        pieces.append(reindexed)

    if not pieces:
        return pd.DataFrame(index=daily_index, columns=["NDVI", "NDWI"], dtype="float64")
    return pd.concat(pieces).sort_index().reindex(daily_index)


def merge_weather(
    power_df: pd.DataFrame,
    smap_df: pd.DataFrame | None = None,
    sentinel_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join all three sources on ``(date, geoid)``, then add derived features.

    SMAP and Sentinel are both optional; pass ``None`` (or an empty frame) to
    skip. Forward-fills Sentinel inside each county so daily rows have a
    usable NDVI/NDWI.
    """
    out = power_df.copy()

    if smap_df is not None and not smap_df.empty:
        out = out.join(smap_df, how="left")

    if sentinel_df is not None and not sentinel_df.empty:
        s_filled = _ffill_sentinel_to_daily(sentinel_df, out.index)
        out = out.join(s_filled, how="left")

    out = compute_gdd(out)
    return out


# ---------------------------------------------------------------------------
# Public per-county / per-county-set fetchers
# ---------------------------------------------------------------------------

def fetch_county_weather(
    geoid: str,
    geometry,
    start_year: int,
    end_year: int,
    parameters: Sequence[str] = ALL_NASA_PARAMS,
    include_smap: bool = True,
    include_sentinel: bool = True,
    county_row=None,
    refresh: bool = False,
    add_rolling: bool = True,
) -> pd.DataFrame:
    """All-in-one weather frame for a single county.

    Returns a DataFrame indexed by ``(date, geoid)``. The same call with the
    same args is byte-for-byte deterministic on a warm cache (every source is
    keyed on ``(geoid, start_year, end_year)`` and writes a parquet on first
    miss).
    """
    power = fetch_county_power(
        geoid=geoid, geometry=geometry,
        start_year=start_year, end_year=end_year,
        parameters=parameters, county_row=county_row, refresh=refresh,
    )

    smap = None
    if include_smap:
        smap = fetch_county_smap(
            geoid=geoid, geometry=geometry,
            start_year=start_year, end_year=end_year,
            county_row=county_row, refresh=refresh,
        )

    sentinel = None
    if include_sentinel:
        sentinel = fetch_county_sentinel(
            geoid=geoid, geometry=geometry,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31",
            refresh=refresh,
        )

    merged = merge_weather(power, smap, sentinel)
    if add_rolling:
        merged = add_rolling_features(merged)
    return merged


def fetch_counties_weather(
    counties,
    start_year: int,
    end_year: int,
    parameters: Sequence[str] = ALL_NASA_PARAMS,
    include_smap: bool = True,
    include_sentinel: bool = True,
    refresh: bool = False,
    add_rolling: bool = True,
    sleep_between: float = 1.0,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Vectorized version over a county GeoDataFrame.

    Caches the final merged frame at
    ``<data_root>/derived/weather/weather_daily_{nrows}_{hash}_{start}_{end}.parquet``
    so repeat calls with the same (county set, date range) are an in-memory
    parquet read — even though the underlying per-county pulls are also cached
    individually, materializing the merge once is ~free and saves the join
    cost on the hot path.

    ``max_workers`` controls parallel NASA POWER + SMAP county fetches (1 =
    legacy sequential + ``sleep_between`` throttling; 4+ recommended for
    cold pulls). Sentinel uses the same cap when included.
    """
    cache = merged_cache_path(
        counties["geoid"], start_year, end_year, suffix="daily",
    )
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    w = max(1, int(max_workers))

    power = fetch_counties_power(
        counties, start_year=start_year, end_year=end_year,
        parameters=parameters, refresh=refresh, sleep_between=sleep_between,
        max_workers=w,
    )

    smap = None
    if include_smap:
        smap = fetch_counties_smap(
            counties, start_year=start_year, end_year=end_year,
            refresh=refresh, sleep_between=sleep_between,
            max_workers=w,
        )

    sentinel = None
    if include_sentinel:
        sentinel = fetch_counties_sentinel(
            counties,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31",
            refresh=refresh,
            max_workers=w,
        )

    merged = merge_weather(power, smap, sentinel)
    if add_rolling:
        merged = add_rolling_features(merged)

    merged.to_parquet(cache)
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _summarize(df: pd.DataFrame, start_year: int, end_year: int) -> str:
    n_geoids = df.index.get_level_values("geoid").nunique()
    n_rows = len(df)
    cols_present = [
        c for c in (
            "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN",
            "GWETROOT", "SMAP_surface_sm_m3m3",
            "NDVI", "NDWI", "GDD", "GDD_cumulative",
        ) if c in df.columns
    ]
    return "\n".join([
        f"period:    {start_year}–{end_year}",
        f"counties:  {n_geoids}",
        f"rows:      {n_rows:,} (date × geoid)",
        f"columns:   {len(df.columns)} ({', '.join(cols_present)}, …)",
    ])


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build/refresh the per-county weather frame "
                    "(NASA POWER + SMAP + Sentinel-2)."
    )
    parser.add_argument("--start", type=int, required=True,
                        help="Start year (inclusive). NASA POWER goes back to 1981; "
                             f"SMAP starts {SMAP_FIRST_YEAR}; Sentinel-2 starts "
                             f"{SENTINEL_FIRST_DATE}. Anything below those just gets "
                             "blanks for the affected source.")
    parser.add_argument("--end", type=int, required=True,
                        help="End year (inclusive).")
    parser.add_argument("--states", nargs="+", default=None, metavar="STATE",
                        help="One or more state names / 2-digit FIPS. Omit for "
                             "all 5 target states.")
    parser.add_argument("--geoid", nargs="+", default=None, metavar="GEOID",
                        help="One or more 5-digit county FIPS to subset to "
                             "(applied after --states).")
    parser.add_argument("--no-smap", action="store_true",
                        help="Skip SMAP soil moisture pulls.")
    parser.add_argument("--no-sentinel", action="store_true",
                        help="Skip Sentinel-2 NDVI/NDWI pulls (often the "
                             "slowest piece by far).")
    parser.add_argument("--no-rolling", action="store_true",
                        help="Skip 7d/30d rolling features. Useful when you "
                             "just want the raw merged frame.")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download from POWER / SMAP / Sentinel and "
                             "rebuild the merged parquet cache.")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to sleep between live POWER calls (skipped "
                             "on cached counties). Default 1.0 — not used when "
                             "--max-fetch-workers is greater than 1.")
    parser.add_argument(
        "--max-fetch-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel county fetches for POWER + SMAP + Sentinel (1 = sequential).",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional output path (.parquet or .csv) for the "
                             "daily frame.")
    parser.add_argument("--annual-out", type=Path, default=None,
                        help="Optional output path (.parquet or .csv) for the "
                             "annual summary frame.")
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error(f"--start ({args.start}) cannot be after --end ({args.end})")

    # Lazy import so just running `--help` doesn't boot geopandas/shapely.
    from engine.counties import load_counties

    counties = load_counties(states=args.states)
    if args.geoid:
        wanted = {str(g) for g in args.geoid}
        counties = counties[counties["geoid"].isin(wanted)].reset_index(drop=True)
        if counties.empty:
            parser.error(f"no counties matched --geoid {args.geoid}")

    df = fetch_counties_weather(
        counties,
        start_year=args.start,
        end_year=args.end,
        include_smap=not args.no_smap,
        include_sentinel=not args.no_sentinel,
        refresh=args.refresh,
        add_rolling=not args.no_rolling,
        sleep_between=args.sleep,
        max_workers=int(args.max_fetch_workers),
    )
    print(_summarize(df, args.start, args.end))

    def _write(frame: pd.DataFrame, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix == ".parquet":
            frame.to_parquet(target)
        elif suffix == ".csv":
            frame.to_csv(target)
        else:
            raise SystemExit(f"unsupported --out suffix: {suffix} (use .parquet or .csv)")
        print(f"wrote {target}", file=sys.stderr)

    if args.out is not None:
        _write(df, args.out)

    if args.annual_out is not None:
        annual = build_annual_summary(df)
        _write(annual, args.annual_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
