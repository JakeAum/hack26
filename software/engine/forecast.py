"""End-to-end deliverable forecast pipeline.

Glues the four engine sources, the trained TFT checkpoints, the analog-year
cone, and the NASS state baseline into a single CLI that emits the artifact
the judges (and the deck) consume:

    per-county forecasts (model cone + analog cone) +
    per-state aggregations (corn-area-weighted, with NASS comparison)

CLI examples::

    # Single forecast date
    hack26-forecast --year 2025 --as-of 2025-08-01 --states Iowa Nebraska \\
                    --model-dir ~/hack26/data/derived/models/final \\
                    --out forecasts_2025_aug1.parquet

    # All four forecast dates in one run (the standard deliverable invocation)
    hack26-forecast --year 2025 --all-dates --model-dir <...> \\
                    --out forecasts_2025.parquet --log-file run_2025.log

The CLI is intentionally tolerant of partial inputs:

- If the per-(forecast_date) checkpoint is missing, that variant is skipped
  with a warning rather than aborting the whole run.
- If the analog cone fails for a county, that county's analog row is dropped
  but the model row is still emitted.
- If the NASS state baseline isn't published yet for ``target_year``, the
  ``nass_baseline_*`` columns are NaN.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ._logging import (
    add_cli_logging_args,
    apply_cli_logging_args,
    banner,
    get_logger,
    log_environment,
)
from .analogs import analog_cones_for_counties
from .dataset import (
    MAX_TRAIN_YEAR,
    MIN_TRAIN_YEAR,
    build_inference_dataset,
)
from .model import (
    FORECAST_DATES,
    load_tft,
    predict_tft,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Calendar mappings
# ---------------------------------------------------------------------------

#: Forecast-date -> calendar cursor (month, day) within the target year.
FORECAST_DATE_CALENDAR: dict[str, tuple[int, int]] = {
    "aug1":  (8, 1),
    "sep1":  (9, 1),
    "oct1":  (10, 1),
    "final": (11, 30),
}

#: Forecast-date -> NASS reference_period_desc string for the state baseline.
FORECAST_DATE_TO_NASS_REF: dict[str, str] = {
    "aug1":  "YEAR - AUG FORECAST",
    "sep1":  "YEAR - SEP FORECAST",
    "oct1":  "YEAR - OCT FORECAST",
    "final": "YEAR - NOV FORECAST",   # Nov forecast is the closest in-season baseline
}


def _as_of_for(year: int, forecast_date: str) -> str:
    m, d = FORECAST_DATE_CALENDAR[forecast_date]
    return f"{int(year):04d}-{m:02d}-{d:02d}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """Area-weighted quantile.

    Uses the midpoint-weight CDF (Hyndman-Fan type 7-equivalent for unit
    weights) so the uniform-weight case matches ``np.percentile(..., 50)``
    exactly. Tail values clamp to min/max.
    """
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v = np.asarray(values, dtype="float64")[mask]
    w = np.asarray(weights, dtype="float64")[mask]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w) - 0.5 * w
    total = w.sum()
    if total <= 0:
        return float(np.median(v))
    cw_norm = cw / total
    return float(np.interp(float(q), cw_norm, v))


def aggregate_county_forecasts_to_state(
    county_forecasts: pd.DataFrame,
    counties_meta: pd.DataFrame,
    cdl_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-county forecasts to per-state, area-weighted by corn area.

    Args:
        county_forecasts: DataFrame with at least
            ``geoid, forecast_date, yield_p10, yield_p50, yield_p90``.
        counties_meta: county catalog with ``geoid, state_fips, state_name``.
        cdl_meta: CDL frame with ``geoid, corn_area_m2`` (one row per county
            for the target year — caller pre-filters / pre-merges).

    Returns:
        DataFrame with one row per ``(state_fips, forecast_date)``:
        ``state_fips, state_name, forecast_date, n_counties, total_corn_area_m2,
        state_yield_p10, state_yield_p50, state_yield_p90``.
    """
    if county_forecasts.empty:
        return pd.DataFrame()

    weights = cdl_meta[["geoid", "corn_area_m2"]].copy()
    weights["geoid"] = weights["geoid"].astype(str)
    df = county_forecasts.merge(weights, on="geoid", how="left")
    df["corn_area_m2"] = df["corn_area_m2"].fillna(0.0)
    df = df.merge(
        counties_meta[["geoid", "state_fips", "state_name"]].astype(
            {"geoid": str, "state_fips": str}
        ),
        on="geoid", how="left",
    )

    out_rows: list[dict] = []
    for (state_fips, fd), grp in df.groupby(["state_fips", "forecast_date"]):
        weights_arr = grp["corn_area_m2"].to_numpy(dtype="float64")
        total_area = float(weights_arr.sum())
        if total_area <= 0:
            # Fallback to unweighted if no corn area info — better than NaN.
            weights_arr = np.ones_like(weights_arr)
            total_area_for_log = 0.0
        else:
            total_area_for_log = total_area

        p10 = _weighted_quantile(grp["yield_p10"].to_numpy("float64"),
                                 weights_arr, 0.5)
        p50 = float(np.average(grp["yield_p50"].to_numpy("float64"),
                               weights=weights_arr))
        p90 = _weighted_quantile(grp["yield_p90"].to_numpy("float64"),
                                 weights_arr, 0.5)

        # Wider, more honest cone: take the area-weighted mean of county P10s
        # (lower bound on lower-bound) and the area-weighted mean of P90s
        # (upper bound on upper-bound). This is what we actually want for the
        # state-level cone — averaging quantiles across counties is the
        # tightest reasonable answer.
        p10_mean = float(np.average(grp["yield_p10"].to_numpy("float64"),
                                    weights=weights_arr))
        p90_mean = float(np.average(grp["yield_p90"].to_numpy("float64"),
                                    weights=weights_arr))

        state_name = grp["state_name"].iloc[0] if "state_name" in grp else ""
        out_rows.append({
            "state_fips": str(state_fips),
            "state_name": str(state_name),
            "forecast_date": str(fd),
            "n_counties": int(len(grp)),
            "total_corn_area_m2": float(total_area_for_log),
            "state_yield_p10": float(p10_mean),
            "state_yield_p50": float(p50),
            "state_yield_p90": float(p90_mean),
        })

    return pd.DataFrame(out_rows).sort_values(["state_fips", "forecast_date"])


def _attach_nass_state_baseline(
    state_forecasts: pd.DataFrame,
    nass_state_df: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    """Add ``nass_baseline_bu_acre`` and the matching reference period to each
    state-level forecast row (one column per forecast date)."""
    if state_forecasts.empty or nass_state_df.empty:
        if not state_forecasts.empty:
            state_forecasts = state_forecasts.copy()
            state_forecasts["nass_baseline_bu_acre"] = float("nan")
            state_forecasts["nass_baseline_ref"] = ""
        return state_forecasts

    out = state_forecasts.copy()
    nass_yr = nass_state_df[nass_state_df["year"].astype(int) == int(target_year)]
    baseline_vals: list[float] = []
    baseline_refs: list[str] = []
    for _, row in out.iterrows():
        ref = FORECAST_DATE_TO_NASS_REF.get(row["forecast_date"], "YEAR")
        match = nass_yr[
            (nass_yr["state_ansi"].astype(str).str.zfill(2) == row["state_fips"])
            & (nass_yr["reference_period_desc"] == ref)
        ]
        if match.empty:
            baseline_vals.append(float("nan"))
            baseline_refs.append("")
        else:
            baseline_vals.append(float(match.iloc[0]["nass_value"]))
            baseline_refs.append(ref)
    out["nass_baseline_bu_acre"] = baseline_vals
    out["nass_baseline_ref"] = baseline_refs
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _resolve_forecast_dates(spec: str) -> list[str]:
    if spec == "all":
        return list(FORECAST_DATES)
    if spec in FORECAST_DATES:
        return [spec]
    raise ValueError(
        f"unknown forecast-date {spec!r}; expected 'all' or one of "
        f"{list(FORECAST_DATES)}"
    )


def _resolve_model_dir(model_dir: str | Path) -> Path:
    p = Path(model_dir).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--model-dir does not exist: {p}")
    return p


def _checkpoint_path(model_dir: Path, forecast_date: str) -> Path:
    return model_dir / f"tft_{forecast_date}.pt"


def _load_models(
    model_dir: Path, forecast_dates: Sequence[str]
) -> dict[str, object]:
    """Load every requested checkpoint; warn (not crash) on missing variants."""
    out: dict[str, object] = {}
    for fd in forecast_dates:
        ckpt = _checkpoint_path(model_dir, fd)
        if not ckpt.exists():
            logger.warning(
                "checkpoint missing for forecast_date=%s (looked at %s) — "
                "skipping that variant",
                fd, ckpt,
            )
            continue
        out[fd] = load_tft(ckpt)
        logger.info("loaded checkpoint:  forecast_date=%s  path=%s", fd, ckpt)
    return out


def _county_corn_area(target_year: int, counties: pd.DataFrame, refresh: bool,
                       allow_download: bool = False) -> pd.DataFrame:
    """Pull a single-year CDL frame so we can area-weight the state agg."""
    from engine.cdl import fetch_counties_cdl
    try:
        df = fetch_counties_cdl(counties, year=int(target_year),
                                resolution=30, refresh=refresh,
                                allow_download=allow_download)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CDL %d unavailable (%s); falling back to %d",
                       target_year, exc, MAX_TRAIN_YEAR)
        df = fetch_counties_cdl(counties, year=MAX_TRAIN_YEAR,
                                resolution=30, refresh=refresh,
                                allow_download=allow_download)
    keep = [c for c in ("geoid", "corn_area_m2") if c in df.columns]
    return df[keep].copy() if keep else pd.DataFrame(
        {"geoid": counties["geoid"].astype(str), "corn_area_m2": 0.0}
    )


def run_forecast(
    target_year: int,
    forecast_dates: Sequence[str],
    states: Sequence[str] | None,
    model_dir: Path,
    history_start: int = MIN_TRAIN_YEAR,
    history_end: int = MAX_TRAIN_YEAR,
    k_analogs: int = 5,
    include_smap: bool = True,
    include_sentinel: bool = False,
    refresh: bool = False,
    num_samples: int = 200,
    allow_download: bool = False,
    max_fetch_workers: int = 4,
) -> dict[str, pd.DataFrame]:
    """Run the full forecast pipeline. Returns a dict with keys:

    - ``county_model``: per-(geoid, forecast_date) TFT cone
    - ``county_analog``: per-(geoid, forecast_date) analog-year cone
    - ``state``: per-(state, forecast_date) area-weighted aggregation +
       NASS baseline comparison
    """
    from engine.counties import load_counties
    from engine.nass import (
        fetch_counties_nass_yields,
        fetch_nass_state_corn_forecasts,
    )

    forecast_dates = list(forecast_dates)
    banner(
        f"FORECAST RUN  year={target_year}  dates={forecast_dates}  "
        f"states={states or 'ALL'}  model_dir={model_dir}",
        logger=logger,
    )

    counties = load_counties(states=states)
    logger.info("counties: n=%d  states=%s",
                len(counties),
                sorted(counties["state_fips"].unique().tolist()))

    w = max(1, int(max_fetch_workers))
    bundle = build_inference_dataset(
        states=states,
        target_year=target_year,
        include_smap=include_smap,
        include_sentinel=include_sentinel,
        refresh=refresh,
        history_start_year=history_start,
        history_end_year=min(history_end, MAX_TRAIN_YEAR),
        allow_download=allow_download,
        max_fetch_workers=w,
    )
    logger.info("inference bundle: n_series=%d", bundle.n_series)

    # ---- model cones ----
    models = _load_models(model_dir, forecast_dates)
    if not models:
        raise FileNotFoundError(
            f"no checkpoints loaded from {model_dir}; nothing to forecast"
        )

    county_model_pieces: list[pd.DataFrame] = []
    for fd, mdl in models.items():
        preds = predict_tft(mdl, bundle, forecast_date=fd,
                            num_samples=num_samples)
        county_model_pieces.append(preds)
    county_model = (
        pd.concat(county_model_pieces, ignore_index=True)
        if county_model_pieces else pd.DataFrame()
    )
    logger.info("county model cone: rows=%d  forecast_dates=%s",
                len(county_model),
                county_model["forecast_date"].unique().tolist()
                if not county_model.empty else [])

    # ---- analog cones (one per forecast date) ----
    history_years = list(
        range(history_start, min(history_end, MAX_TRAIN_YEAR) + 1)
    )
    weather_cache: pd.DataFrame | None = None
    yields_cache: pd.DataFrame | None = None
    analog_pieces: list[pd.DataFrame] = []

    for fd in forecast_dates:
        as_of = _as_of_for(target_year, fd)
        # Reuse weather/yields across the four cone runs.
        if weather_cache is None:
            from engine.weather import fetch_counties_weather
            weather_cache = fetch_counties_weather(
                counties,
                start_year=min([target_year, *history_years]),
                end_year=max([target_year, *history_years]),
                include_smap=include_smap,
                include_sentinel=include_sentinel,
                refresh=refresh,
                max_workers=w,
            )
        if yields_cache is None:
            yields_cache = fetch_counties_nass_yields(
                counties,
                start_year=min(history_years),
                end_year=max(history_years),
                refresh=refresh,
                max_workers=w,
            )
        a_df = analog_cones_for_counties(
            counties=counties,
            target_year=target_year,
            as_of=as_of,
            history_years=history_years,
            k=k_analogs,
            weather=weather_cache,
            yields=yields_cache,
        )
        a_df["forecast_date"] = fd
        analog_pieces.append(a_df)

    county_analog = (
        pd.concat(analog_pieces, ignore_index=True)
        if analog_pieces else pd.DataFrame()
    )
    logger.info("county analog cone: rows=%d", len(county_analog))

    # ---- state aggregation ----
    cdl_meta = _county_corn_area(target_year, counties, refresh=refresh,
                                  allow_download=allow_download)
    state_df = aggregate_county_forecasts_to_state(
        county_forecasts=county_model,
        counties_meta=counties,
        cdl_meta=cdl_meta,
    )
    if not state_df.empty and not county_analog.empty:
        analog_state_pieces: list[pd.DataFrame] = []
        df = county_analog.merge(
            cdl_meta, on="geoid", how="left",
        ).merge(
            counties[["geoid", "state_fips", "state_name"]].astype(
                {"geoid": str, "state_fips": str}
            ),
            on="geoid", how="left",
        )
        df["corn_area_m2"] = df["corn_area_m2"].fillna(0.0)
        for (state_fips, fd), grp in df.groupby(["state_fips", "forecast_date"]):
            w = grp["corn_area_m2"].to_numpy("float64")
            if w.sum() <= 0:
                w = np.ones_like(w)
            row = {
                "state_fips": str(state_fips),
                "forecast_date": str(fd),
                "analog_min":   float(np.average(grp["yield_min"], weights=w)),
                "analog_p25":   float(np.average(grp["yield_p25"], weights=w)),
                "analog_p50":   float(np.average(grp["yield_p50"], weights=w)),
                "analog_p75":   float(np.average(grp["yield_p75"], weights=w)),
                "analog_max":   float(np.average(grp["yield_max"], weights=w)),
                "analog_mean":  float(np.average(grp["yield_mean"], weights=w)),
            }
            analog_state_pieces.append(pd.DataFrame([row]))
        if analog_state_pieces:
            astate = pd.concat(analog_state_pieces, ignore_index=True)
            state_df = state_df.merge(
                astate, on=["state_fips", "forecast_date"], how="left",
            )

    # ---- NASS baseline ----
    try:
        nass_state = fetch_nass_state_corn_forecasts(
            state_fips_list=sorted(counties["state_fips"].unique().tolist()),
            start_year=int(target_year), end_year=int(target_year),
            refresh=refresh,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("NASS state baseline pull failed (%s); leaving NaN", exc)
        nass_state = pd.DataFrame()
    state_df = _attach_nass_state_baseline(state_df, nass_state, target_year)

    # Add as-of column for clarity in the deliverable.
    if not county_model.empty:
        county_model = county_model.copy()
        county_model["as_of"] = county_model["forecast_date"].map(
            lambda fd: _as_of_for(target_year, fd)
        )
    if not state_df.empty:
        state_df = state_df.copy()
        state_df["as_of"] = state_df["forecast_date"].map(
            lambda fd: _as_of_for(target_year, fd)
        )
        state_df["target_year"] = int(target_year)

    return {
        "county_model": county_model,
        "county_analog": county_analog,
        "state": state_df,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(path, index=False)
    elif suf == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output suffix: {suf} (use .parquet or .csv)")
    logger.info("wrote %s  rows=%d  bytes=%d", path, len(df), path.stat().st_size)


# ---------------------------------------------------------------------------
# CLI: hack26-forecast
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deliverable corn-yield forecast for a target year."
    )
    parser.add_argument("--year", type=int, required=True,
                        help="Target year (typically 2025).")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all-dates", action="store_true",
                     help="Run all four forecast dates (aug1/sep1/oct1/final).")
    grp.add_argument("--forecast-date", default=None,
                     choices=list(FORECAST_DATES),
                     help="Run a single forecast-date variant.")
    parser.add_argument("--as-of", default=None,
                        help="Override the calendar cursor for the single "
                             "forecast-date run (ISO YYYY-MM-DD). Default is "
                             "derived from --forecast-date.")
    parser.add_argument("--states", nargs="+", default=None, metavar="STATE",
                        help="Subset of states (names or 2-digit FIPS). "
                             "Omit for all 5.")
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing tft_{aug1,sep1,oct1,final}.pt "
                             "checkpoints.")
    parser.add_argument("--history-start", type=int, default=MIN_TRAIN_YEAR,
                        help=f"Earliest analog-history year (default {MIN_TRAIN_YEAR}).")
    parser.add_argument("--history-end", type=int, default=MAX_TRAIN_YEAR,
                        help=f"Latest analog-history year (default {MAX_TRAIN_YEAR}; "
                             f"clamped to {MAX_TRAIN_YEAR} by 2025-leak guard).")
    parser.add_argument("--k-analogs", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Quantile samples per series at predict-time.")
    parser.add_argument("--include-sentinel", action="store_true")
    parser.add_argument("--no-smap", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-download", action="store_true",
                        help="Permit CDL raster downloads from USDA when a "
                             "year is missing from the data root. Without "
                             "this flag missing CDL years are skipped.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Primary output path for the per-state result. "
                             "Sibling files are written for county-level model "
                             "and analog cones.")
    parser.add_argument(
        "--max-fetch-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel I/O for weather/NASS during inference and analog prep.",
    )
    add_cli_logging_args(parser)
    args = parser.parse_args(argv)

    log_path = apply_cli_logging_args(args, tag="forecast")
    log_environment(logger)
    logger.info("rotated log: %s", log_path)
    logger.info("argv: %s", " ".join(sys.argv))

    if args.all_dates:
        forecast_dates = list(FORECAST_DATES)
    else:
        forecast_dates = [args.forecast_date]

    try:
        model_dir = _resolve_model_dir(args.model_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 2

    try:
        result = run_forecast(
            target_year=int(args.year),
            forecast_dates=forecast_dates,
            states=args.states,
            model_dir=model_dir,
            history_start=int(args.history_start),
            history_end=int(args.history_end),
            k_analogs=int(args.k_analogs),
            include_smap=not args.no_smap,
            include_sentinel=args.include_sentinel,
            refresh=args.refresh,
            num_samples=int(args.num_samples),
            allow_download=args.allow_download,
            max_fetch_workers=int(args.max_fetch_workers),
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("forecast aborted: %s", exc)
        return 2

    state_df = result["state"]
    county_model = result["county_model"]
    county_analog = result["county_analog"]

    # Pretty per-state log dump — judges-friendly summary in the run log.
    if not state_df.empty:
        banner(f"DELIVERABLE SUMMARY  year={args.year}", logger=logger)
        for _, r in state_df.iterrows():
            logger.info(
                "  %s (%s) %s as_of=%s  P50=%.1f  cone[P10,P90]=[%.1f,%.1f]  "
                "analog[min,P50,max]=[%.1f,%.1f,%.1f]  NASS_baseline=%s",
                r.get("state_name", "?"), r.get("state_fips", ""),
                r["forecast_date"], r.get("as_of", "?"),
                float(r["state_yield_p50"]),
                float(r["state_yield_p10"]), float(r["state_yield_p90"]),
                float(r.get("analog_min", float("nan"))),
                float(r.get("analog_p50", float("nan"))),
                float(r.get("analog_max", float("nan"))),
                f"{float(r['nass_baseline_bu_acre']):.1f}"
                if pd.notna(r.get("nass_baseline_bu_acre"))
                else "—",
            )

    # Output files.
    if args.out is not None:
        out = Path(args.out).expanduser().resolve()
        _write_output(state_df, out)
        county_path = out.with_name(out.stem + "_per_county_model" + out.suffix)
        analog_path = out.with_name(out.stem + "_per_county_analog" + out.suffix)
        _write_output(county_model, county_path)
        _write_output(county_analog, analog_path)
    else:
        with pd.option_context("display.max_rows", 40, "display.width", 220):
            print(state_df, file=sys.stderr)

    logger.info("forecast pipeline complete. log: %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
