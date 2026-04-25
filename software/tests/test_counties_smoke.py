"""Smoke test: pull all counties, slice to the first 5 in Colorado, verify shape.

Runs the real download once (cached after that), so the first invocation does
hit the Census TIGER server. Designed to fail loudly with a readable summary
before we point this at a cloud worker.

Usage:
    python -m tests.test_counties_smoke      # standalone, prints a report
    pytest tests/test_counties_smoke.py      # CI-style
"""

from __future__ import annotations

import sys

from engine.counties import TARGET_STATES, load_counties

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


def _crs_label(crs) -> str:
    """Compact CRS label; pyproj's str(crs) is a huge PROJJSON dump."""
    if crs is None:
        return "<none>"
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    name = getattr(crs, "name", "?")
    return f"EPSG:{epsg} ({name})" if epsg else name


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
