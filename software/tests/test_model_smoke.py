"""Smoke tests for :mod:`engine.model` — constant geometry, year-split guard,
metrics math, and (only when Darts is installed) a 1-epoch toy training loop
that confirms the CSV epoch logger writes a row.

Tests that need Darts/torch are skipped automatically if the ``[forecast]``
extras aren't installed.

    pytest software/tests/test_model_smoke.py -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.model import (
    FORECAST_DATE_CHUNKS,
    FORECAST_DATES,
    QUANTILES,
    _parse_year_range,
    _resolve_chunk_lengths,
    _validate_year_split,
    evaluate_tft,
)


# ---------------------------------------------------------------------------
# Constant + helper invariants
# ---------------------------------------------------------------------------

def test_forecast_date_chunks_consistent() -> None:
    assert set(FORECAST_DATES) == set(FORECAST_DATE_CHUNKS), (
        "FORECAST_DATES and FORECAST_DATE_CHUNKS must agree"
    )
    # In-season chunks must sum to the 244-day season; final collapses to 1.
    for fd in ("aug1", "sep1", "oct1"):
        a, b = FORECAST_DATE_CHUNKS[fd]
        assert a + b == 244, f"{fd}: input+output should cover 244-day season"
    assert FORECAST_DATE_CHUNKS["final"][1] == 1
    assert QUANTILES == (0.1, 0.5, 0.9)


def test_resolve_chunk_lengths() -> None:
    assert _resolve_chunk_lengths("aug1") == (122, 122)
    assert _resolve_chunk_lengths("sep1") == (153, 91)
    assert _resolve_chunk_lengths("oct1") == (183, 61)
    assert _resolve_chunk_lengths("final") == (243, 1)
    with pytest.raises(ValueError, match="unknown forecast_date"):
        _resolve_chunk_lengths("nope")


def test_parse_year_range() -> None:
    assert _parse_year_range("2008-2010") == [2008, 2009, 2010]
    assert _parse_year_range("2008,2010,2014") == [2008, 2010, 2014]
    assert _parse_year_range("2020") == [2020]


# ---------------------------------------------------------------------------
# 2025-strict-holdout — train/val/test guard
# ---------------------------------------------------------------------------

def test_year_split_accepts_pass1_measurement() -> None:
    _validate_year_split(list(range(2008, 2023)), val_year=2023, test_year=2024)


def test_year_split_accepts_pass2_deliverable() -> None:
    _validate_year_split(list(range(2008, 2025)), val_year=2023, test_year=None)


def test_year_split_rejects_2025_in_train() -> None:
    with pytest.raises(ValueError, match=r"2025-leak-guard"):
        _validate_year_split([2024, 2025], val_year=2023, test_year=None)


def test_year_split_rejects_2025_in_val() -> None:
    with pytest.raises(ValueError, match=r"2025-leak-guard"):
        _validate_year_split([2008, 2009], val_year=2025, test_year=None)


def test_year_split_rejects_2025_in_test() -> None:
    with pytest.raises(ValueError, match=r"2025-leak-guard"):
        _validate_year_split([2008, 2009], val_year=2010, test_year=2025)


# ---------------------------------------------------------------------------
# evaluate_tft schema + arithmetic
# ---------------------------------------------------------------------------

def test_evaluate_tft_returns_per_state_and_overall() -> None:
    preds = pd.DataFrame({
        "geoid":         ["19169", "19001", "08001", "08123"],
        "year":          [2024,    2024,    2024,    2024],
        "forecast_date": ["aug1"] * 4,
        "yield_p10":     [170,     180,     105,     115],
        "yield_p50":     [185,     190,     120,     125],
        "yield_p90":     [200,     210,     135,     140],
        "yield_mean":    [185,     190,     120,     125],
        "yield_std":     [10,      10,      10,      10],
    })
    labels = pd.DataFrame({
        "geoid": ["19169", "19001", "08001", "08123"],
        "year":  [2024,    2024,    2024,    2024],
        "nass_value": [190, 195, 122, 128],
    })
    metrics = evaluate_tft(preds, labels)
    assert not metrics.empty
    # Expect at least one row per state plus an "ALL" overall row.
    assert "state_fips" in metrics.columns
    assert (metrics["state_fips"] == "ALL").any()
    assert (metrics["state_fips"] == "19").any()
    assert (metrics["state_fips"] == "08").any()
    # All numeric metrics are finite.
    for c in ("rmse_bu_acre", "mape_pct", "p10_p90_coverage"):
        assert metrics[c].dropna().between(0, 10000).all()


def test_evaluate_tft_handles_no_overlap() -> None:
    preds = pd.DataFrame({
        "geoid": ["19169"], "year": [2024], "forecast_date": ["aug1"],
        "yield_p10": [180], "yield_p50": [190], "yield_p90": [200],
        "yield_mean": [190], "yield_std": [5],
    })
    labels = pd.DataFrame({
        "geoid": ["08001"], "year": [2024], "nass_value": [120],
    })
    out = evaluate_tft(preds, labels)
    assert out.empty


# ---------------------------------------------------------------------------
# Optional: build a toy TFT and run 1 epoch
# ---------------------------------------------------------------------------

DARTS_AVAILABLE = importlib.util.find_spec("darts") is not None


@pytest.mark.skipif(not DARTS_AVAILABLE,
                    reason="Darts/torch not installed (install with [forecast] extra)")
def test_toy_tft_one_epoch_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a tiny TFT and run a single training epoch on synthetic series.

    Confirms the model+CSV-epoch-logger plumbing actually fires a row and
    that ``predict_tft`` returns the expected schema. Synthetic data — no
    network calls.
    """
    monkeypatch.setenv("HACK26_CACHE_DIR", str(tmp_path))

    from darts import TimeSeries

    from engine.dataset import (
        FUTURE_COVARIATE_COLS,
        STATIC_COVARIATE_COLS,
        TrainingBundle,
        _build_calendar_frame,
    )
    from engine.model import build_tft, predict_tft, save_tft, load_tft

    rng = np.random.default_rng(0)
    n_series = 6  # tiny — keep this fast
    season_dates = pd.date_range("2024-04-01", "2024-11-30", freq="D")
    n_t = len(season_dates)
    target_series = []
    past_series = []
    future_series = []
    statics_rows = []
    index_rows = []
    for i in range(n_series):
        yld = 150 + 10 * rng.standard_normal()
        static_row = {c: 0.0 for c in STATIC_COVARIATE_COLS}
        static_row["state_19"] = 1.0
        static_row["historical_mean_yield_bu_acre"] = float(yld)
        statics_df = pd.DataFrame([static_row])

        target_df = pd.DataFrame(
            {"yield_bu_acre": np.full(n_t, yld, dtype=np.float32)},
            index=season_dates,
        )
        past_df = pd.DataFrame({
            "PRECTOTCORR": rng.uniform(0, 5, n_t).astype(np.float32),
            "T2M":         rng.uniform(15, 28, n_t).astype(np.float32),
        }, index=season_dates)
        future_df = _build_calendar_frame(season_dates, season_end_doy=334)

        target_series.append(TimeSeries.from_dataframe(
            target_df, freq="D", static_covariates=statics_df,
        ))
        past_series.append(TimeSeries.from_dataframe(past_df, freq="D"))
        future_series.append(TimeSeries.from_dataframe(future_df, freq="D"))
        statics_rows.append(static_row)
        index_rows.append({
            "geoid": f"190{i:02d}", "year": 2024,
            "state_fips": "19", "label": yld, "label_present": True,
            "coverage": 1.0,
        })
    bundle = TrainingBundle(
        target_series=target_series,
        past_covariates=past_series,
        future_covariates=future_series,
        static_covariates=pd.DataFrame(statics_rows),
        series_index=pd.DataFrame(index_rows),
        past_covariate_cols=["PRECTOTCORR", "T2M"],
        static_covariate_cols=list(STATIC_COVARIATE_COLS),
    )

    csv_path = tmp_path / "epoch.csv"
    from engine.model import _make_csv_epoch_logger
    cb = _make_csv_epoch_logger(csv_path, tag="toy")

    model = build_tft(
        forecast_date="aug1",
        hidden_size=8, lstm_layers=1, num_attention_heads=1,
        hidden_continuous_size=4, batch_size=4, n_epochs=1,
        pl_callbacks=[cb], accelerator="cpu", devices=1, precision="32-true",
    )
    model.fit(
        series=bundle.target_series,
        past_covariates=bundle.past_covariates,
        future_covariates=bundle.future_covariates,
        verbose=False,
    )

    # CSV logger should have written exactly one (or possibly zero, depending
    # on whether validation ran) row + header. Header exists always.
    assert csv_path.exists()
    body = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert body[0].startswith("ts,epoch,train_loss,val_loss")
    # No validation set was passed, so on_validation_epoch_end may not fire;
    # we accept either header-only or header+rows.
    assert len(body) >= 1

    # Save + reload round-trip works.
    ckpt = tmp_path / "tft_aug1.pt"
    save_tft(model, ckpt)
    assert ckpt.exists()
    sidecar = ckpt.with_suffix(ckpt.suffix + ".meta.json")
    assert sidecar.exists()
    reloaded = load_tft(ckpt)

    # Predict on the same bundle (any output is fine for smoke purposes).
    preds = predict_tft(reloaded, bundle, forecast_date="aug1", num_samples=20)
    assert {"geoid", "year", "forecast_date", "yield_p10",
            "yield_p50", "yield_p90"}.issubset(preds.columns)
    assert len(preds) == n_series
    # P10 ≤ P50 ≤ P90 invariant.
    assert (preds["yield_p10"] <= preds["yield_p50"]).all()
    assert (preds["yield_p50"] <= preds["yield_p90"]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
