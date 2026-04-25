"""Smoke test: pull a tiny weather frame for one Iowa county and verify the
SPEC §2 contract — ``(date, geoid)`` index, NASA POWER columns present, GDD
computed, and a *second* call returns a byte-identical frame off the parquet
cache (the determinism / consistency guarantee).

Hits the live NASA POWER API once on the first run (a few hundred KB), then
all subsequent runs are local parquet reads. Sentinel-2 is skipped to keep
the test fast and offline-friendly.

Usage:
    pytest software/tests/test_weather_smoke.py     # CI-style
    python -m tests.test_weather_smoke              # standalone, prints a report
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from engine.counties import load_counties
from engine.weather import (
    ALL_NASA_PARAMS,
    NASA_TEMP_PARAMS,
    fetch_county_power,
    fetch_county_weather,
)
from engine.weather._cache import power_cache_path

# Story County, Iowa — Iowa State University's home county, stable signal.
STORY_COUNTY_GEOID = "19169"
START_YEAR = 2022
END_YEAR = 2022


def _story_row():
    iowa = load_counties(states=["Iowa"])
    matches = iowa[iowa["geoid"] == STORY_COUNTY_GEOID]
    assert not matches.empty, f"Story County ({STORY_COUNTY_GEOID}) missing from catalog"
    return matches.iloc[0]


@pytest.mark.skipif(
    sys.platform.startswith("win") and not power_cache_path(
        STORY_COUNTY_GEOID, START_YEAR, END_YEAR,
    ).exists(),
    reason="cold NASA POWER pull from CI is flaky; pre-warm cache or run locally.",
)
def test_story_county_power_consistency() -> None:
    """Two back-to-back calls for the same (geoid, range) must be identical —
    that's the lookup-consistency guarantee."""
    row = _story_row()

    df1 = fetch_county_power(
        geoid=row["geoid"], geometry=row.geometry,
        start_year=START_YEAR, end_year=END_YEAR,
        parameters=NASA_TEMP_PARAMS, county_row=row,
    )
    df2 = fetch_county_power(
        geoid=row["geoid"], geometry=row.geometry,
        start_year=START_YEAR, end_year=END_YEAR,
        parameters=NASA_TEMP_PARAMS, county_row=row,
    )

    assert df1.index.names == ["date", "geoid"], df1.index.names
    assert (df1.index.get_level_values("geoid") == STORY_COUNTY_GEOID).all()
    assert "T2M_MAX" in df1.columns and "T2M_MIN" in df1.columns
    assert len(df1) >= 360, f"expected ~365 daily rows, got {len(df1)}"

    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.skipif(
    not power_cache_path(STORY_COUNTY_GEOID, START_YEAR, END_YEAR).exists(),
    reason="needs a warmed POWER cache (run test_story_county_power_consistency first).",
)
def test_full_county_weather_has_gdd() -> None:
    """End-to-end ``fetch_county_weather`` (SMAP on, Sentinel off) yields the
    full frame plus derived GDD columns."""
    row = _story_row()

    df = fetch_county_weather(
        geoid=row["geoid"], geometry=row.geometry,
        start_year=START_YEAR, end_year=END_YEAR,
        include_smap=True, include_sentinel=False,
        county_row=row, add_rolling=True,
    )

    assert df.index.names == ["date", "geoid"]
    assert "GDD" in df.columns, "compute_gdd should have run"
    assert "GDD_cumulative" in df.columns
    # Iowa corn season: cumulative GDD over a year should be in the high
    # hundreds to low thousands of °C-days; sanity-check the order of magnitude.
    end_year_max_cum = df["GDD_cumulative"].max()
    assert 500 <= end_year_max_cum <= 4000, (
        f"GDD_cumulative max {end_year_max_cum} outside sane Iowa range"
    )
    # Rolling features should exist for at least one base column.
    assert any(c.endswith("_7d_avg") for c in df.columns)


def _main() -> int:
    print("[smoke] loading Story County, IA …")
    row = _story_row()
    print(f"[smoke] geoid={row['geoid']}  centroid=({row['centroid_lat']:.3f},"
          f"{row['centroid_lon']:.3f})")
    print(f"[smoke] fetching {START_YEAR}–{END_YEAR} NASA POWER (temp params only)…")
    df = fetch_county_power(
        geoid=row["geoid"], geometry=row.geometry,
        start_year=START_YEAR, end_year=END_YEAR,
        parameters=NASA_TEMP_PARAMS, county_row=row,
    )
    print(f"[smoke] rows: {len(df)}  cols: {list(df.columns)}")
    print(df.head(3).to_string())
    try:
        test_story_county_power_consistency()
        test_full_county_weather_has_gdd()
    except AssertionError as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr)
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
