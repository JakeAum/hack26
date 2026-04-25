"""Smoke tests for :mod:`engine.forecast` — the calendar map, the
weighted-quantile aggregation, the per-state rollup, and the NASS-baseline
attachment.

Offline. No network calls. (The end-to-end ``run_forecast`` integration test
needs trained checkpoints + cached weather and is out of scope for offline
smoke testing — covered by the deliverable run on the AWS box instead.)

    pytest software/tests/test_forecast_smoke.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from engine.forecast import (
    FORECAST_DATE_CALENDAR,
    FORECAST_DATE_TO_NASS_REF,
    _as_of_for,
    _attach_nass_state_baseline,
    _resolve_forecast_dates,
    _weighted_quantile,
    aggregate_county_forecasts_to_state,
)
from engine.model import FORECAST_DATES


# ---------------------------------------------------------------------------
# Calendar mapping
# ---------------------------------------------------------------------------

def test_forecast_date_calendar_covers_all_variants() -> None:
    assert set(FORECAST_DATE_CALENDAR) == set(FORECAST_DATES)
    assert set(FORECAST_DATE_TO_NASS_REF) == set(FORECAST_DATES)


def test_as_of_for_2025() -> None:
    assert _as_of_for(2025, "aug1") == "2025-08-01"
    assert _as_of_for(2025, "sep1") == "2025-09-01"
    assert _as_of_for(2025, "oct1") == "2025-10-01"
    assert _as_of_for(2025, "final") == "2025-11-30"


def test_resolve_forecast_dates() -> None:
    assert _resolve_forecast_dates("all") == list(FORECAST_DATES)
    assert _resolve_forecast_dates("aug1") == ["aug1"]
    with pytest.raises(ValueError, match="unknown forecast-date"):
        _resolve_forecast_dates("nope")


# ---------------------------------------------------------------------------
# Weighted-quantile primitive
# ---------------------------------------------------------------------------

def test_weighted_quantile_uniform_weights_match_unweighted() -> None:
    v = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    w = np.ones_like(v)
    assert pytest.approx(_weighted_quantile(v, w, 0.5), abs=1.0) == 120.0
    # Tails clamp to min/max
    assert _weighted_quantile(v, w, 0.0) == 100.0
    assert _weighted_quantile(v, w, 1.0) == 140.0


def test_weighted_quantile_skews_with_weights() -> None:
    v = np.array([100.0, 200.0])
    # Weight the second point 9x more — median should sit near 200.
    out = _weighted_quantile(v, np.array([1.0, 9.0]), 0.5)
    assert out > 150.0


def test_weighted_quantile_handles_empty() -> None:
    assert np.isnan(_weighted_quantile(np.array([]), np.array([]), 0.5))


# ---------------------------------------------------------------------------
# Per-state aggregation
# ---------------------------------------------------------------------------

def _mock_county_forecasts() -> pd.DataFrame:
    return pd.DataFrame({
        "geoid":         ["19169", "19001", "08001", "08123"],
        "year":          [2025] * 4,
        "forecast_date": ["aug1"] * 4,
        "yield_p10":     [170.0, 180.0, 105.0, 115.0],
        "yield_p50":     [185.0, 190.0, 120.0, 125.0],
        "yield_p90":     [200.0, 210.0, 135.0, 140.0],
    })


def _mock_counties_meta() -> pd.DataFrame:
    return pd.DataFrame({
        "geoid":      ["19169", "19001", "08001", "08123"],
        "state_fips": ["19", "19", "08", "08"],
        "state_name": ["Iowa", "Iowa", "Colorado", "Colorado"],
    })


def _mock_cdl_meta() -> pd.DataFrame:
    return pd.DataFrame({
        "geoid":        ["19169", "19001", "08001", "08123"],
        "corn_area_m2": [1.0e9, 5.0e8, 2.0e8, 5.0e7],
    })


def test_aggregation_area_weights_p50_correctly() -> None:
    state_df = aggregate_county_forecasts_to_state(
        county_forecasts=_mock_county_forecasts(),
        counties_meta=_mock_counties_meta(),
        cdl_meta=_mock_cdl_meta(),
    )
    assert not state_df.empty
    assert set(state_df["state_fips"]) == {"19", "08"}
    iowa = state_df[state_df["state_fips"] == "19"].iloc[0]
    co = state_df[state_df["state_fips"] == "08"].iloc[0]

    # Iowa P50 = (185*1e9 + 190*0.5e9) / 1.5e9 = ~186.66
    expected_iowa = (185.0 * 1.0e9 + 190.0 * 0.5e9) / (1.5e9)
    assert pytest.approx(iowa["state_yield_p50"], rel=1e-4) == expected_iowa
    # Colorado: (120*2e8 + 125*5e7) / 2.5e8 = 121.0
    expected_co = (120.0 * 2.0e8 + 125.0 * 5.0e7) / (2.5e8)
    assert pytest.approx(co["state_yield_p50"], rel=1e-4) == expected_co

    # Cone bookends are sane
    assert iowa["state_yield_p10"] < iowa["state_yield_p50"] < iowa["state_yield_p90"]
    assert co["state_yield_p10"] < co["state_yield_p50"] < co["state_yield_p90"]
    assert iowa["n_counties"] == 2
    assert co["n_counties"] == 2
    assert iowa["total_corn_area_m2"] == 1.5e9


def test_aggregation_handles_zero_corn_area_falls_back_to_unweighted() -> None:
    cdl = _mock_cdl_meta().copy()
    cdl["corn_area_m2"] = 0.0  # no corn anywhere — fallback should kick in
    state_df = aggregate_county_forecasts_to_state(
        county_forecasts=_mock_county_forecasts(),
        counties_meta=_mock_counties_meta(),
        cdl_meta=cdl,
    )
    assert not state_df.empty
    iowa = state_df[state_df["state_fips"] == "19"].iloc[0]
    # Falls back to plain mean of P50 (no NaNs).
    assert pytest.approx(iowa["state_yield_p50"], rel=1e-4) == np.mean([185.0, 190.0])
    assert iowa["total_corn_area_m2"] == 0.0


# ---------------------------------------------------------------------------
# NASS baseline attachment
# ---------------------------------------------------------------------------

def test_attach_nass_baseline_matches_reference_period() -> None:
    state_df = pd.DataFrame({
        "state_fips": ["19", "19", "08", "08"],
        "state_name": ["Iowa", "Iowa", "Colorado", "Colorado"],
        "forecast_date": ["aug1", "sep1", "aug1", "sep1"],
        "n_counties": [99] * 4,
        "total_corn_area_m2": [1.0] * 4,
        "state_yield_p10": [180.0, 182.0, 110.0, 112.0],
        "state_yield_p50": [190.0, 191.0, 120.0, 122.0],
        "state_yield_p90": [200.0, 201.0, 130.0, 132.0],
    })
    nass = pd.DataFrame({
        "state_ansi": ["19", "19", "08", "08"],
        "year":       [2025] * 4,
        "reference_period_desc": [
            "YEAR - AUG FORECAST", "YEAR - SEP FORECAST",
            "YEAR - AUG FORECAST", "YEAR - SEP FORECAST",
        ],
        "nass_value": [205.0, 198.0, 145.0, 142.0],
    })
    out = _attach_nass_state_baseline(state_df, nass, target_year=2025)
    assert "nass_baseline_bu_acre" in out.columns
    iowa_aug = out.query("state_fips == '19' and forecast_date == 'aug1'").iloc[0]
    assert iowa_aug["nass_baseline_bu_acre"] == 205.0
    assert iowa_aug["nass_baseline_ref"] == "YEAR - AUG FORECAST"


def test_attach_nass_baseline_missing_baseline_yields_nan() -> None:
    state_df = pd.DataFrame({
        "state_fips": ["19"], "state_name": ["Iowa"],
        "forecast_date": ["aug1"], "n_counties": [99],
        "total_corn_area_m2": [1.0],
        "state_yield_p10": [180.0], "state_yield_p50": [190.0], "state_yield_p90": [200.0],
    })
    out = _attach_nass_state_baseline(state_df, pd.DataFrame(), target_year=2025)
    assert np.isnan(out["nass_baseline_bu_acre"].iloc[0])
    assert out["nass_baseline_ref"].iloc[0] == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
