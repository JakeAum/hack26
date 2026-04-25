"""Tests for :mod:`engine.nass` (Quick Stats) — no network unless ``NASS_API_KEY`` is set.

Unit tests use private helpers. Optional live test hits NASS for one row.

    pytest software/tests/test_nass_smoke.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

from engine.nass.core import (
    _geoid_from_row,
    _is_other_counties_row,
    _normalize_county_yields,
    _normalize_state_forecasts,
    _parse_nass_value,
    _validate_year_range,
    nass_api_key,
)


def test_parse_nass_value() -> None:
    assert _parse_nass_value("1,234.5") == 1234.5
    assert _parse_nass_value("(D)") is None
    assert _parse_nass_value("(X)") is None
    assert _parse_nass_value("200") == 200.0


def test_geoid_from_row() -> None:
    assert _geoid_from_row({"state_ansi": "19", "county_ansi": "169"}) == "19169"
    assert _geoid_from_row({"state_ansi": "8", "county_ansi": "14"}) == "08014"


def test_other_counties_excluded() -> None:
    assert _is_other_counties_row(
        {"county_name": "OTHER COUNTIES", "county_code": "998"}
    )
    assert not _is_other_counties_row(
        {"county_name": "STORY", "county_code": "169"}
    )


def test_normalize_county_yields() -> None:
    raw: list[dict] = [
        {
            "state_ansi": "19",
            "county_ansi": "169",
            "county_name": "STORY",
            "county_code": "169",
            "state_alpha": "IA",
            "state_name": "IOWA",
            "year": 2020,
            "Value": "200.5",
            "reference_period_desc": "YEAR",
            "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
            "statisticcat_desc": "YIELD",
            "load_time": "2021-01-01 00:00:00",
        },
        {
            "state_ansi": "19",
            "county_ansi": "998",
            "county_name": "OTHER (COMBINED) COUNTIES",
            "county_code": "998",
            "year": 2020,
            "Value": "100",
            "reference_period_desc": "YEAR",
        },
    ]
    df = _normalize_county_yields(raw, {"19169"})
    assert len(df) == 1
    assert df["geoid"].iloc[0] == "19169"
    assert df["nass_value"].iloc[0] == 200.5
    assert df["agg_level_desc"].iloc[0] == "COUNTY"


def test_nass_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NASS_API_KEY", raising=False)
    with pytest.raises(OSError, match="NASS_API_KEY"):
        nass_api_key()


def test_validate_year_range() -> None:
    _validate_year_range(2020, 2020)
    _validate_year_range(2018, 2024)
    with pytest.raises(ValueError, match="start_year=2025 > end_year=2024"):
        _validate_year_range(2025, 2024)


def test_normalize_state_forecasts_filters_reference_periods() -> None:
    raw: list[dict] = [
        {
            "state_ansi": "19",
            "agg_level_desc": "STATE",
            "state_alpha": "IA",
            "state_name": "IOWA",
            "year": 2024,
            "reference_period_desc": "YEAR - AUG FORECAST",
            "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
            "statisticcat_desc": "YIELD",
            "Value": "204.3",
            "load_time": "2024-08-12 12:00:00",
        },
        {
            "state_ansi": "19",
            "agg_level_desc": "STATE",
            "state_alpha": "IA",
            "state_name": "IOWA",
            "year": 2024,
            "reference_period_desc": "YEAR - JUL FORECAST",
            "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
            "statisticcat_desc": "YIELD",
            "Value": "201.0",
            "load_time": "2024-07-12 12:00:00",
        },
    ]
    df = _normalize_state_forecasts(raw, "19")
    assert len(df) == 1
    assert df["reference_period_desc"].iloc[0] == "YEAR - AUG FORECAST"
    assert df["geoid"].iloc[0] == "19000"
    assert df["agg_level_desc"].iloc[0] == "STATE"
    assert df["nass_value"].iloc[0] == 204.3


@pytest.mark.skipif(
    not (os.environ.get("NASS_API_KEY") or "").strip(),
    reason="NASS_API_KEY not set; optional live NASS test.",
)
def test_live_iowa_county_2020() -> None:
    from engine.nass import fetch_county_nass_yields

    nass_api_key()  # noqa: the skip guard already validated env
    df = fetch_county_nass_yields("19169", start_year=2020, end_year=2020, refresh=True)
    assert not df.empty, "expected 2020 Story County, IA final yield on Quick Stats"
    assert df["nass_value"].iloc[0] > 50
    assert df["reference_period_desc"].iloc[0] == "YEAR"


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v"] + (sys.argv[1:] or []))
    )
