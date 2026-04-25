"""Smoke test: pull all counties, slice to the first 5 in Colorado, verify shape.

Runs the real download once (cached after that), so the first invocation does
hit the Census TIGER server. Designed to fail loudly with a readable summary
before we point this at a cloud worker.

Usage:
    python -m tests.test_counties_smoke      # standalone, prints a report
    pytest tests/test_counties_smoke.py      # CI-style
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from engine.counties import TARGET_STATES, _crs_label, _resolve_states, load_counties

CO_FIPS = "08"
EXPECTED_COLS = {
    "geoid", "state_fips", "county_fips", "name", "name_full", "state_name",
    "centroid_lat", "centroid_lon", "land_area_m2", "water_area_m2", "geometry",
}
# Generous Colorado bbox (lat 36.9–41.1, lon -109.1– -102.0). Catches centroids
# that are wildly off (e.g. swapped lat/lon, wrong CRS).
CO_LAT_RANGE = (36.5, 41.5)
CO_LON_RANGE = (-109.5, -101.5)


def test_first_five_colorado_counties() -> None:
    gdf = load_counties(states=["Colorado"])

    # 1. Schema contract.
    missing = EXPECTED_COLS - set(gdf.columns)
    assert not missing, f"missing expected columns: {missing}"

    # 2. Filter actually filtered.
    assert gdf["state_fips"].eq(CO_FIPS).all(), "non-Colorado rows leaked through"
    assert gdf["state_name"].eq(TARGET_STATES[CO_FIPS]).all()

    # 3. Colorado has 64 counties; sanity-check the total.
    assert 60 <= len(gdf) <= 70, f"unexpected CO county count: {len(gdf)}"

    head = gdf.head(5).reset_index(drop=True)
    assert len(head) == 5, "expected 5 sample rows"

    # 4. Per-row invariants on the sample.
    for i, row in head.iterrows():
        assert row["geoid"].startswith(CO_FIPS), f"row {i} geoid {row['geoid']!r} not in CO"
        assert len(row["geoid"]) == 5, f"row {i} geoid wrong length"
        assert row["name"], f"row {i} missing name"
        assert row["geometry"] is not None, f"row {i} missing geometry"
        assert row["geometry"].is_valid, f"row {i} geometry invalid"
        assert CO_LAT_RANGE[0] <= row["centroid_lat"] <= CO_LAT_RANGE[1], (
            f"row {i} ({row['name']}) centroid_lat {row['centroid_lat']} outside CO"
        )
        assert CO_LON_RANGE[0] <= row["centroid_lon"] <= CO_LON_RANGE[1], (
            f"row {i} ({row['name']}) centroid_lon {row['centroid_lon']} outside CO"
        )
        assert row["land_area_m2"] is not None and row["land_area_m2"] > 0


def test_empty_states_raises() -> None:
    """Regression: empty --states (or load_counties(states=[])) must fail loudly,
    not silently return zero counties."""
    with pytest.raises(ValueError, match="empty"):
        _resolve_states([])
    with pytest.raises(ValueError, match="empty"):
        load_counties(states=[])


def test_cli_empty_states_flag_errors_cleanly() -> None:
    """Regression: `python -m engine.counties --states` (no values) must
    exit non-zero with a usage error, not silently print 0 counties."""
    proc = subprocess.run(
        [sys.executable, "-m", "engine.counties", "--states"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0, (
        f"expected non-zero exit; got {proc.returncode}\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    assert "counties total: 0" not in proc.stdout, "regression: silent zero-row output"


def _main() -> int:
    print("[smoke] loading counties (first call may download ~120MB)...")
    gdf = load_counties(states=["Colorado"])
    print(f"[smoke] CRS:           {_crs_label(gdf.crs)}")
    print(f"[smoke] CO counties:   {len(gdf)}")
    print(f"[smoke] columns:       {list(gdf.columns)}")
    print()
    print("[smoke] first 5 Colorado counties:")
    cols = ["geoid", "name_full", "centroid_lat", "centroid_lon", "land_area_m2"]
    print(gdf.head(5)[cols].to_string(index=False))
    print()

    try:
        test_first_five_colorado_counties()
    except AssertionError as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr)
        return 1

    print("[smoke] PASS - schema, filter, and per-row invariants all hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
