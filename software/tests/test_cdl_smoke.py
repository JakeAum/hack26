"""Smoke test: aggregate the national CDL raster down to the first 5 Iowa
counties and sanity-check the per-county corn numbers.

Skipped automatically if rasterio isn't installed (local dev box) or if no
national CDL raster is reachable on disk (i.e. we'd otherwise trigger a 1.6 GB
download just to run a smoke test). Designed to *actually run* on the AWS
sagemaker workshop machine where ``~/hack26/data/2025_10m_cdls.tif`` is
already mounted from EFS.

Usage:
    pytest software/tests/test_cdl_smoke.py        # CI-style
    python -m tests.test_cdl_smoke                 # standalone, prints a report
"""

from __future__ import annotations

import sys

import pytest

from engine.cdl import (
    LATEST_YEAR,
    _find_existing_tif,
    _tif_basename,
)

# Iowa: corn country. Pick a year/resolution combo that's likely to be on disk
# already. Order: 2025 10m (workshop EFS), 2025 30m, 2024 30m.
CANDIDATES: list[tuple[int, int]] = [
    (2025, 10),
    (2025, 30),
    (2024, 30),
    (LATEST_YEAR, 30),
]
IOWA_FIPS = "19"
# Iowa's statewide corn share is ~30-35% of total county area in a typical year.
# These bounds are intentionally loose so the test catches "completely wrong"
# (e.g. lat/lon swapped, CRS mismatch, masking returning whole raster) without
# being a regression magnet against year-to-year corn acreage shifts.
CORN_PCT_LO = 0.05   # 5% — low even for a poorly-sampled western Iowa county
CORN_PCT_HI = 0.85   # 85% — a checked-out Story-county-style monoculture ceiling


def _pick_available() -> tuple[int, int] | None:
    """Find the first (year, resolution) pair whose .tif is already on disk.

    ``_find_existing_tif`` raises ``FileNotFoundError`` when the data root
    itself is missing — caught here so the test skips cleanly on machines
    without the EFS mount instead of erroring out at collection time.
    """
    for year, res in CANDIDATES:
        try:
            if _find_existing_tif(year, res) is not None:
                return (year, res)
        except FileNotFoundError:
            return None
    return None


def _data_root_for_skip_msg() -> str:
    """Best-effort string for the skip message; never raises."""
    try:
        from engine.cdl import _data_root
        return str(_data_root())
    except FileNotFoundError as e:
        return str(e).splitlines()[0]


pytest.importorskip("rasterio", reason="CDL smoke test needs rasterio")


def test_iowa_first_five_counties_corn() -> None:
    pick = _pick_available()
    if pick is None:
        pytest.skip(
            "no pre-extracted CDL raster found in the data root; engine is "
            "in strict mode and won't trigger a 1.6+ GB download in a smoke "
            f"test.\ndata root: {_data_root_for_skip_msg()}\n"
            f"expected one of: {[_tif_basename(y, r) for y, r in CANDIDATES]}"
        )
    year, resolution = pick

    from engine.cdl import fetch_counties_cdl
    from engine.counties import load_counties

    counties = load_counties(states=["Iowa"]).head(5).reset_index(drop=True)
    assert len(counties) == 5, "expected 5 Iowa sample counties"

    df = fetch_counties_cdl(counties, year=year, resolution=resolution)

    expected_cols = {
        "geoid", "year", "resolution_m", "pixel_area_m2",
        "total_pixels", "cropland_pixels",
        "corn_pixels", "corn_area_m2", "corn_pct_of_county", "corn_pct_of_cropland",
        "soybean_pixels", "soybean_area_m2",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"missing columns: {missing}"

    assert (df["geoid"].str[:2] == IOWA_FIPS).all(), "non-Iowa rows leaked through"
    assert (df["resolution_m"] == resolution).all()
    assert (df["pixel_area_m2"] == resolution * resolution).all(), (
        f"pixel area wrong: expected {resolution**2} m², got "
        f"{df['pixel_area_m2'].unique()}"
    )

    for _, row in df.iterrows():
        assert row["total_pixels"] > 0, f"{row['geoid']}: zero pixels (mask failed?)"
        assert row["corn_pixels"] >= 0
        assert row["corn_pixels"] <= row["total_pixels"], (
            f"{row['geoid']}: corn_pixels {row['corn_pixels']} > "
            f"total_pixels {row['total_pixels']}"
        )
        # Iowa is corn country — every county should have *some* corn.
        assert row["corn_pixels"] > 0, f"{row['geoid']}: zero corn pixels in Iowa?"
        assert CORN_PCT_LO <= row["corn_pct_of_county"] <= CORN_PCT_HI, (
            f"{row['geoid']}: corn_pct_of_county={row['corn_pct_of_county']:.3f} "
            f"outside expected Iowa range [{CORN_PCT_LO}, {CORN_PCT_HI}]"
        )


def _main() -> int:
    pick = _pick_available()
    if pick is None:
        print(f"[smoke] no CDL raster on disk; data root: {_data_root_for_skip_msg()}",
              file=sys.stderr)
        return 0
    year, resolution = pick
    print(f"[smoke] using CDL {year} @ {resolution} m: {_find_existing_tif(year, resolution)}")

    from engine.cdl import fetch_counties_cdl
    from engine.counties import load_counties

    counties = load_counties(states=["Iowa"]).head(5).reset_index(drop=True)
    df = fetch_counties_cdl(counties, year=year, resolution=resolution)
    cols = ["geoid", "corn_pixels", "corn_area_m2", "corn_pct_of_county", "corn_pct_of_cropland"]
    print(df[cols].to_string(index=False))

    try:
        test_iowa_first_five_counties_corn()
    except AssertionError as e:
        print(f"[smoke] FAIL: {e}", file=sys.stderr)
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
