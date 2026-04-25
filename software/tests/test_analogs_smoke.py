"""Smoke tests for :mod:`engine.analogs` — deterministic K-NN over a
handcrafted fixture frame so we can assert exact analog years.

Offline. No network calls.

    pytest software/tests/test_analogs_smoke.py -q
"""

from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine.analogs import (
    DEFAULT_K,
    _doy_for_year,
    _parse_as_of,
    analog_cone,
    analog_cones_for_counties,
    season_to_date_signature,
)


# ---------------------------------------------------------------------------
# Helpers under test
# ---------------------------------------------------------------------------

def test_parse_as_of_string_and_date() -> None:
    assert _parse_as_of("2025-08-01") == date(2025, 8, 1)
    assert _parse_as_of(date(2025, 9, 1)) == date(2025, 9, 1)


def test_doy_for_year_re_anchors_calendar() -> None:
    # Aug 1 in 2024 is DOY 214 (leap year), in 2023 it's 213.
    assert _doy_for_year(2024, date(2025, 8, 1)) == 214
    assert _doy_for_year(2023, date(2025, 8, 1)) == 213


# ---------------------------------------------------------------------------
# Fixture: a synthetic 5-year weather frame for one county
# ---------------------------------------------------------------------------

def _make_synthetic_weather() -> pd.DataFrame:
    """Build a multi-year (date, geoid)-indexed weather frame with hand-tuned
    cumulative GDD / precip so the K-NN result is predictable."""
    rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(42)
    # Five history years + one target year. Target year (2024) is intentionally
    # close to 2018 in cumulative-GDD signature.
    profiles = {
        2018: {"gdd_per_day": 18.0, "rain_per_day": 5.0},
        2019: {"gdd_per_day": 16.0, "rain_per_day": 4.0},
        2020: {"gdd_per_day": 14.0, "rain_per_day": 3.0},
        2021: {"gdd_per_day": 12.0, "rain_per_day": 2.5},
        2022: {"gdd_per_day": 10.0, "rain_per_day": 2.0},
        2023: {"gdd_per_day": 8.0,  "rain_per_day": 1.5},
        2024: {"gdd_per_day": 17.9, "rain_per_day": 5.05},  # ~2018 lookalike
    }
    for yr, prof in profiles.items():
        dates = pd.date_range(f"{yr}-04-01", f"{yr}-11-30", freq="D")
        n = len(dates)
        gdd = np.full(n, prof["gdd_per_day"], dtype="float64") + rng.normal(0, 0.5, n)
        gdd = np.clip(gdd, 0, None)
        df = pd.DataFrame(
            {
                "PRECTOTCORR": np.full(n, prof["rain_per_day"], dtype="float64")
                + rng.normal(0, 0.1, n),
                "GDD": gdd,
                "GDD_cumulative": np.cumsum(gdd),
                "NDVI": np.linspace(0.2, 0.85, n),
            },
            index=dates,
        )
        df = df.assign(geoid="19169").set_index("geoid", append=True)
        df.index = df.index.set_names(["date", "geoid"])
        rows.append(df)
    return pd.concat(rows).sort_index()


def _make_synthetic_yields() -> pd.DataFrame:
    """A NASS-style frame with realized yield per (geoid, year)."""
    return pd.DataFrame({
        "geoid": ["19169"] * 6,
        "year":  [2018, 2019, 2020, 2021, 2022, 2023],
        "nass_value": [200, 190, 175, 165, 145, 130],
    })


# ---------------------------------------------------------------------------
# season_to_date_signature
# ---------------------------------------------------------------------------

def test_signature_uses_cumulative_gdd_through_cutoff() -> None:
    weather = _make_synthetic_weather()
    sig = season_to_date_signature(
        weather, geoid="19169", year=2018, as_of="2018-08-01",
    )
    # GDD is roughly 18/day from Apr 1 to Aug 1 = ~122 days ⇒ ~2196.
    assert 1800 < sig["gdd_cumulative"] < 2600
    # Precip: ~5 mm/day * 122 days = ~610 mm
    assert 500 < sig["precip_cumulative_mm"] < 700
    # NDVI mean is finite and in [0, 1].
    assert 0 < sig["ndvi_mean"] < 1


def test_signature_returns_nans_for_missing_geoid() -> None:
    weather = _make_synthetic_weather()
    sig = season_to_date_signature(
        weather, geoid="99999", year=2020, as_of="2020-08-01",
    )
    assert all(np.isnan(v) for v in sig.values())


# ---------------------------------------------------------------------------
# analog_cone — deterministic K-NN
# ---------------------------------------------------------------------------

def test_analog_cone_picks_2018_first_for_2024_lookalike() -> None:
    weather = _make_synthetic_weather()
    yields = _make_synthetic_yields()
    res = analog_cone(
        geoid="19169", target_year=2024, as_of="2024-08-01",
        history_years=[2018, 2019, 2020, 2021, 2022, 2023],
        weather=weather, yields=yields, k=3,
    )
    # 2024 was tuned to look like 2018 — that should be the nearest analog.
    assert res.analog_years[0] == 2018
    # The K=3 neighbors should be the warmest/wettest history years (2018, 2019, 2020).
    assert set(res.analog_years[:3]) == {2018, 2019, 2020}
    # Yields are read off the synthetic NASS frame.
    assert res.analog_yields[0] == 200.0
    # Empirical summary is internally consistent.
    assert res.yield_min <= res.yield_p25 <= res.yield_p50 <= res.yield_p75 <= res.yield_max
    assert res.k == 3
    # Features used always include cumulative GDD + precip in this fixture.
    assert "gdd_cumulative" in res.features_used
    assert "precip_cumulative_mm" in res.features_used


def test_analog_cone_dict_round_trip_serializable() -> None:
    weather = _make_synthetic_weather()
    yields = _make_synthetic_yields()
    res = analog_cone(
        geoid="19169", target_year=2024, as_of="2024-08-01",
        history_years=[2018, 2019, 2020, 2021, 2022, 2023],
        weather=weather, yields=yields, k=DEFAULT_K,
    )
    d = res.to_dict()
    # All values JSON-friendly (no numpy scalars left).
    import json
    json.dumps(d)


def test_analog_cone_raises_when_too_few_history_years() -> None:
    weather = _make_synthetic_weather()
    yields = _make_synthetic_yields()
    with pytest.raises(ValueError, match=r"k="):
        analog_cone(
            geoid="19169", target_year=2024, as_of="2024-08-01",
            history_years=[2018, 2019],   # fewer than k=5
            weather=weather, yields=yields, k=5,
        )


# ---------------------------------------------------------------------------
# 2025-leak guard for the batch entry point
# ---------------------------------------------------------------------------

def test_batch_rejects_history_years_past_2024() -> None:
    counties = pd.DataFrame({
        "geoid": ["19169"], "state_fips": ["19"],
    })
    with pytest.raises(ValueError, match=r"2025-leak-guard"):
        analog_cones_for_counties(
            counties, target_year=2025, as_of="2025-08-01",
            history_years=[2023, 2024, 2025],  # 2025 is forbidden in history
            k=2,
            weather=pd.DataFrame(), yields=pd.DataFrame(),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
