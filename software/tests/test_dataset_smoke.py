"""Smoke tests for :mod:`engine.dataset` — covers the 2025-leak guard, the
``TrainingBundle.filter_by_year`` machinery, and the static-row builder.

All tests here are offline — no network calls into POWER / NASS / CDL.

    pytest software/tests/test_dataset_smoke.py -q
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from engine.dataset import (
    BUNDLE_META_VERSION,
    DEFAULT_SEASON_END_DOY,
    DEFAULT_SEASON_START_DOY,
    FUTURE_COVARIATE_COLS,
    MAX_TRAIN_YEAR,
    MIN_TRAIN_YEAR,
    STATE_FIPS_INDEX,
    STATE_ONEHOT_COLS,
    STATIC_COVARIATE_COLS,
    TrainingBundle,
    _build_calendar_frame,
    _build_static_row,
    _historical_mean_yields,
    _impute_past_block,
    _validate_train_year_range,
    load_training_bundle,
    load_training_bundle_meta,
    save_training_bundle,
    training_bundle_fits_train_request,
)


# ---------------------------------------------------------------------------
# Year-range guard (2025-strict-holdout)
# ---------------------------------------------------------------------------

def test_train_year_range_accepts_2008_2024() -> None:
    _validate_train_year_range(MIN_TRAIN_YEAR, MAX_TRAIN_YEAR)
    _validate_train_year_range(2010, 2020)


def test_train_year_range_rejects_2025_holdout() -> None:
    with pytest.raises(ValueError, match=r"2025-strict-holdout"):
        _validate_train_year_range(MIN_TRAIN_YEAR, 2025)


def test_train_year_range_rejects_2026_or_later() -> None:
    with pytest.raises(ValueError, match=r"MAX_TRAIN_YEAR"):
        _validate_train_year_range(MIN_TRAIN_YEAR, 2030)


def test_train_year_range_rejects_too_early() -> None:
    with pytest.raises(ValueError, match=r"MIN_TRAIN_YEAR"):
        _validate_train_year_range(1990, MAX_TRAIN_YEAR)


def test_train_year_range_rejects_inverted() -> None:
    with pytest.raises(ValueError, match=r"cannot be after"):
        _validate_train_year_range(2020, 2015)


# ---------------------------------------------------------------------------
# Static row + one-hot encoding
# ---------------------------------------------------------------------------

def test_state_onehot_covers_all_target_states() -> None:
    assert len(STATE_ONEHOT_COLS) == 5
    for fips in STATE_FIPS_INDEX:
        assert f"state_{fips}" in STATE_ONEHOT_COLS
    # All onehot cols are part of the static schema.
    for col in STATE_ONEHOT_COLS:
        assert col in STATIC_COVARIATE_COLS


def test_build_static_row_iowa() -> None:
    county_row = pd.Series({
        "geoid": "19169",
        "state_fips": "19",
        "centroid_lat": 42.0,
        "centroid_lon": -93.5,
        "land_area_m2": 1.5e9,
    })
    cdl_row = pd.Series({
        "geoid": "19169",
        "year": 2022,
        "corn_pct_of_county": 0.55,
        "corn_pct_of_cropland": 0.45,
        "corn_area_m2": 8.0e8,
        "soybean_pixels": 1_000_000,
        "cropland_pixels": 4_000_000,
    })
    row = _build_static_row(
        geoid="19169", year=2022,
        county_row=county_row, cdl_row=cdl_row,
        historical_mean=185.0,
    )
    # Iowa is "19" -> state_19 = 1.0, others = 0.0
    assert row["state_19"] == 1.0
    for fips in STATE_FIPS_INDEX:
        if fips != "19":
            assert row[f"state_{fips}"] == 0.0
    assert row["corn_pct_of_county"] == 0.55
    assert row["corn_pct_of_cropland"] == 0.45
    assert pytest.approx(row["soybean_pct_of_cropland"], abs=1e-6) == 0.25
    assert row["centroid_lat"] == 42.0
    assert row["historical_mean_yield_bu_acre"] == 185.0
    # Every column declared in the schema should be present.
    for col in STATIC_COVARIATE_COLS:
        assert col in row


def test_build_static_row_handles_missing_cdl() -> None:
    county_row = pd.Series({
        "geoid": "08123", "state_fips": "08",
        "centroid_lat": 40.0, "centroid_lon": -104.0,
        "land_area_m2": 5e9,
    })
    row = _build_static_row(
        geoid="08123", year=2010,
        county_row=county_row, cdl_row=None,
        historical_mean=120.0,
    )
    assert row["state_08"] == 1.0
    assert row["corn_pct_of_county"] == 0.0  # CDL absent -> zeroed crop fractions
    assert row["historical_mean_yield_bu_acre"] == 120.0


# ---------------------------------------------------------------------------
# Calendar (future-covariate) builder
# ---------------------------------------------------------------------------

def test_calendar_frame_shape_and_columns() -> None:
    dates = pd.date_range("2024-04-01", "2024-11-30", freq="D")
    df = _build_calendar_frame(dates, season_end_doy=DEFAULT_SEASON_END_DOY)
    assert len(df) == len(dates)
    for col in FUTURE_COVARIATE_COLS:
        assert col in df.columns
    # sin/cos sanity: should be in [-1, 1]
    assert df["doy_sin"].between(-1, 1).all()
    assert df["doy_cos"].between(-1, 1).all()
    # days_until_end_of_season monotonically decreases
    diffs = df["days_until_end_of_season"].diff().dropna()
    assert (diffs <= 0).all()


# ---------------------------------------------------------------------------
# Past-covariate imputation
# ---------------------------------------------------------------------------

def test_impute_past_block_fills_holes() -> None:
    dates = pd.date_range("2024-04-01", periods=10, freq="D")
    raw = pd.DataFrame({
        "PRECTOTCORR": [1.0, np.nan, np.nan, 2.0, 3.0, np.nan, 4.0, np.nan, np.nan, 5.0],
        "T2M":         [10.0, 11.0, np.nan, 12.0, np.nan, 13.0, 14.0, np.nan, 15.0, 16.0],
    }, index=dates)
    out = _impute_past_block(raw, ["PRECTOTCORR", "T2M"])
    assert out.shape == (10, 2)
    assert not out.isna().any().any()
    # ffill/bfill semantics: leading NaN filled with first observed.
    assert float(out["PRECTOTCORR"].iloc[1]) == 1.0
    assert float(out["T2M"].iloc[2]) == 11.0


def test_impute_past_block_handles_all_nan_column() -> None:
    dates = pd.date_range("2024-04-01", periods=5, freq="D")
    raw = pd.DataFrame({
        "X": [np.nan] * 5,
        "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
    }, index=dates)
    out = _impute_past_block(raw, ["X", "Y"])
    assert out.isna().sum().sum() == 0
    assert (out["X"] == 0.0).all()


# ---------------------------------------------------------------------------
# Historical-mean yields (leak-free helper)
# ---------------------------------------------------------------------------

def test_historical_mean_strictly_prior_years() -> None:
    yields = pd.DataFrame({
        "geoid": ["19169"] * 5 + ["08001"] * 4,
        "year":  [2018, 2019, 2020, 2021, 2022, 2018, 2019, 2020, 2021],
        "nass_value": [180, 190, 200, 175, 195, 110, 120, 130, 125],
    })
    means = _historical_mean_yields(yields, target_year=2022)
    # Only 2018-2021 are < 2022.
    assert pytest.approx(means["19169"]) == np.mean([180, 190, 200, 175])
    assert pytest.approx(means["08001"]) == np.mean([110, 120, 130, 125])

    means2025 = _historical_mean_yields(yields, target_year=2025)
    assert pytest.approx(means2025["19169"]) == np.mean([180, 190, 200, 175, 195])


def test_historical_mean_handles_no_prior_history() -> None:
    yields = pd.DataFrame({
        "geoid": ["19169"], "year": [2022], "nass_value": [200.0],
    })
    means = _historical_mean_yields(yields, target_year=2022)
    assert means.empty


# ---------------------------------------------------------------------------
# TrainingBundle.filter_by_year
# ---------------------------------------------------------------------------

def test_filter_by_year_lockstep() -> None:
    # We don't have Darts at test time; mock the lists with simple objects
    # and just check that filter_by_year keeps everything in lockstep.
    n = 6
    bundle = TrainingBundle(
        target_series=[f"tgt_{i}" for i in range(n)],
        past_covariates=[f"past_{i}" for i in range(n)],
        future_covariates=[f"fut_{i}" for i in range(n)],
        static_covariates=pd.DataFrame(
            [{c: float(i) for c in STATIC_COVARIATE_COLS} for i in range(n)]
        ),
        series_index=pd.DataFrame({
            "geoid":     ["19001", "19001", "19169", "19169", "08001", "08001"],
            "year":      [2020,    2021,    2020,    2021,    2020,    2021],
            "state_fips": ["19", "19", "19", "19", "08", "08"],
            "label":     [180,     190,     200,     210,     120,     130],
            "label_present": [True] * 6,
            "coverage": [1.0] * 6,
        }),
        past_covariate_cols=["PRECTOTCORR", "T2M"],
        static_covariate_cols=list(STATIC_COVARIATE_COLS),
    )
    sub = bundle.filter_by_year([2020])
    assert sub.n_series == 3
    assert sub.series_index["year"].tolist() == [2020, 2020, 2020]
    # static covariates row count drops to 3 too
    assert len(sub.static_covariates) == 3
    # the lists are in lockstep
    assert sub.target_series == ["tgt_0", "tgt_2", "tgt_4"]
    assert sub.past_covariates == ["past_0", "past_2", "past_4"]
    # past_cov_cols / static_cols are *copied*, not shared references
    assert sub.past_covariate_cols == ["PRECTOTCORR", "T2M"]
    assert sub.past_covariate_cols is not bundle.past_covariate_cols


# ---------------------------------------------------------------------------
# On-disk training bundle (hack26-dataset → hack26-train)
# ---------------------------------------------------------------------------

def _sample_meta() -> dict:
    return {
        "bundle_meta_version": BUNDLE_META_VERSION,
        "states_fips": ["19"],
        "start_year": 2008,
        "end_year": 2024,
        "include_sentinel": False,
        "include_smap": True,
        "season_start_doy": DEFAULT_SEASON_START_DOY,
        "season_end_doy": DEFAULT_SEASON_END_DOY,
    }


def test_training_bundle_fits_train_request_ok() -> None:
    """Cached bundle years/states must cover the train/val/test split."""
    idx = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024],
        "state_fips": ["19"] * 5,
    })
    bundle = TrainingBundle(series_index=idx)
    ok, err = training_bundle_fits_train_request(
        bundle,
        _sample_meta(),
        states_fips=["19"],
        required_years={2020, 2023, 2024},
        include_sentinel=False,
        include_smap=True,
    )
    assert ok and err == ""


def test_training_bundle_fits_train_request_rejects_missing_year() -> None:
    idx = pd.DataFrame({
        "year": [2020, 2021, 2022],
        "state_fips": ["19", "19", "19"],
    })
    bundle = TrainingBundle(series_index=idx)
    ok, err = training_bundle_fits_train_request(
        bundle,
        _sample_meta(),
        states_fips=["19"],
        required_years={2023},
        include_sentinel=False,
        include_smap=True,
    )
    assert not ok
    assert "missing year" in err


def test_save_load_training_bundle_roundtrip(tmp_path) -> None:
    idx = pd.DataFrame({"year": [2022], "state_fips": ["19"]})
    bundle = TrainingBundle(
        target_series=[],
        past_covariates=[],
        future_covariates=[],
        series_index=idx,
    )
    p = tmp_path / "b.pkl"
    save_training_bundle(
        bundle,
        p,
        states_fips=["19"],
        start_year=2018,
        end_year=2022,
        include_sentinel=False,
        include_smap=True,
    )
    assert load_training_bundle_meta(p) is not None
    out = load_training_bundle(p)
    assert out.n_series == 0
    assert out.series_index["year"].tolist() == [2022]


# ---------------------------------------------------------------------------
# Public CLI-flag invariants
# ---------------------------------------------------------------------------

def test_constants_consistency() -> None:
    assert MAX_TRAIN_YEAR == 2024, "2025-strict-holdout: MAX_TRAIN_YEAR must remain 2024"
    assert MIN_TRAIN_YEAR <= MAX_TRAIN_YEAR
    assert DEFAULT_SEASON_START_DOY < DEFAULT_SEASON_END_DOY
    assert (DEFAULT_SEASON_END_DOY - DEFAULT_SEASON_START_DOY + 1) == 244, (
        "season window changed; the model chunk geometry assumes 244 days"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
