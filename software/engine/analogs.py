"""Analog-year cone — empirical uncertainty from historical look-alike years.

Companion to :mod:`engine.model`'s native quantile regression. Where the TFT
gives a model-derived ``[P10, P50, P90]`` from its learned likelihood, this
module produces an **empirical** uncertainty band by asking the question the
problem statement explicitly calls out:

    "What years in the historical record had season-to-date weather most
     similar to this year, and what did corn actually do in those years?"

Method:

1. For one ``(geoid, target_year, as_of_date)`` request, build the
   season-to-date weather signature (cumulative GDD, cumulative precipitation,
   mean NDVI when present).
2. Build the same signature for every ``history_year`` in the requested set.
3. Z-score each signature feature using the history (so units cancel) and
   compute Euclidean distance from target to each history year.
4. Pick the ``K`` (default 5) nearest history years and look up their realized
   NASS final yields.
5. Return the analog years + per-year yields + empirical
   ``min / P25 / P50 / P75 / max / mean / std`` of those yields.

This module deliberately makes **no live API calls**. It pulls everything
through the cached frames produced by :mod:`engine.weather` and
:mod:`engine.nass`, so a forecast run that already warmed those caches can
compute analog cones in seconds for the whole 5-state catalog.

Public surface:

    SIGNATURE_BASE_FEATURES               tuple of feature ids
    season_to_date_signature(weather, geoid, year, as_of)  -> dict
    analog_cone(geoid, target_year, as_of, history_years, k=5, ...) -> dict
    analog_cones_for_counties(counties, target_year, as_of, history_years, ...) -> DataFrame
    _main(argv)                           CLI:  hack26-analogs ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ._logging import (
    StepCounter,
    add_cli_logging_args,
    apply_cli_logging_args,
    banner,
    get_logger,
    log_environment,
)
from .dataset import (
    DEFAULT_SEASON_END_DOY,
    DEFAULT_SEASON_START_DOY,
    MAX_TRAIN_YEAR,
    MIN_TRAIN_YEAR,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Signature feature spec
# ---------------------------------------------------------------------------

#: Names of the season-to-date features the analog matcher uses. The K-NN
#: distance is Euclidean over the **z-scored** version of these. NDVI is
#: included only if every year in the candidate history has it (Sentinel-2
#: starts at 2017 in our merge layer, so most pre-2017 history years drop it).
SIGNATURE_BASE_FEATURES: tuple[str, ...] = (
    "gdd_cumulative",
    "precip_cumulative_mm",
    "ndvi_mean",
    "soil_moisture_mean",
)

DEFAULT_K = 5

# ---------------------------------------------------------------------------
# As-of parsing
# ---------------------------------------------------------------------------

def _parse_as_of(as_of: str | _date) -> _date:
    """Accept ``"2025-08-01"`` (or any ISO ``YYYY-MM-DD``) and return a date."""
    if isinstance(as_of, _date):
        return as_of
    return _date.fromisoformat(str(as_of).strip())


def _doy_for_year(year: int, ref: _date) -> int:
    """DOY of the target ``year``'s analog of the reference month/day.

    Lets us re-use the same calendar position (e.g. "Aug 1") across years
    without bumping into leap-year issues — we always read the matching
    DOY of each history year's growing season.
    """
    return _date(int(year), ref.month, ref.day).timetuple().tm_yday


# ---------------------------------------------------------------------------
# Signature builders
# ---------------------------------------------------------------------------

def season_to_date_signature(
    weather: pd.DataFrame,
    geoid: str,
    year: int,
    as_of: str | _date,
    season_start_doy: int = DEFAULT_SEASON_START_DOY,
) -> dict[str, float]:
    """Compute the season-to-date analog signature for one (geoid, year).

    Args:
        weather: merged daily weather frame (indexed by ``(date, geoid)``)
            with at least ``PRECTOTCORR`` and either ``GDD_cumulative`` or
            ``GDD``. Optionally ``NDVI`` and SMAP/GWET soil moisture.
        geoid: 5-digit county FIPS.
        year: year whose season-to-date window we summarize.
        as_of: cutoff date — features are accumulated through this date,
            inclusive. The month/day of ``as_of`` is re-applied to ``year``
            so you can use one "Aug 1" cursor across many years.
        season_start_doy: growing-season start DOY (default Apr 1 / 91).

    Returns:
        Dict with keys from :data:`SIGNATURE_BASE_FEATURES` (NaN for any
        feature whose source column wasn't present).
    """
    target_ref = _parse_as_of(as_of)
    cutoff_doy = _doy_for_year(year, target_ref)

    if weather.empty:
        return {k: float("nan") for k in SIGNATURE_BASE_FEATURES}

    try:
        block = weather.xs(geoid, level="geoid", drop_level=False)
    except KeyError:
        return {k: float("nan") for k in SIGNATURE_BASE_FEATURES}

    dates = block.index.get_level_values("date")
    yr_mask = dates.year == int(year)
    doy = dates.dayofyear
    in_window = (doy >= season_start_doy) & (doy <= cutoff_doy)
    sub = block[yr_mask & in_window]
    if sub.empty:
        return {k: float("nan") for k in SIGNATURE_BASE_FEATURES}

    out: dict[str, float] = {}

    # Cumulative GDD — prefer the precomputed column from engine.weather, else
    # accumulate raw GDD. Fall back to NaN if neither is present.
    if "GDD_cumulative" in sub.columns and sub["GDD_cumulative"].notna().any():
        out["gdd_cumulative"] = float(sub["GDD_cumulative"].dropna().iloc[-1])
    elif "GDD" in sub.columns:
        out["gdd_cumulative"] = float(sub["GDD"].fillna(0.0).sum())
    else:
        out["gdd_cumulative"] = float("nan")

    out["precip_cumulative_mm"] = (
        float(sub["PRECTOTCORR"].fillna(0.0).sum())
        if "PRECTOTCORR" in sub.columns else float("nan")
    )

    out["ndvi_mean"] = (
        float(sub["NDVI"].dropna().mean())
        if "NDVI" in sub.columns and sub["NDVI"].notna().any()
        else float("nan")
    )

    sm_cols = [c for c in ("SMAP_surface_sm_m3m3", "GWETROOT", "GWETTOP")
               if c in sub.columns]
    if sm_cols:
        # Pick the first non-empty SMAP/GWET column we have for this slice.
        for c in sm_cols:
            if sub[c].notna().any():
                out["soil_moisture_mean"] = float(sub[c].dropna().mean())
                break
        else:
            out["soil_moisture_mean"] = float("nan")
    else:
        out["soil_moisture_mean"] = float("nan")

    return out


@dataclass
class AnalogResult:
    """One analog-cone result (one (geoid, target_year, as_of) query)."""

    geoid: str
    target_year: int
    as_of: str
    k: int
    analog_years: list[int]
    analog_distances: list[float]
    analog_yields: list[float]
    yield_min: float
    yield_p25: float
    yield_p50: float
    yield_p75: float
    yield_max: float
    yield_mean: float
    yield_std: float
    features_used: list[str]

    def to_dict(self) -> dict:
        return {
            "geoid": self.geoid,
            "target_year": self.target_year,
            "as_of": self.as_of,
            "k": self.k,
            "analog_years": list(self.analog_years),
            "analog_distances": [round(float(d), 4) for d in self.analog_distances],
            "analog_yields": [round(float(y), 2) for y in self.analog_yields],
            "yield_min": float(self.yield_min),
            "yield_p25": float(self.yield_p25),
            "yield_p50": float(self.yield_p50),
            "yield_p75": float(self.yield_p75),
            "yield_max": float(self.yield_max),
            "yield_mean": float(self.yield_mean),
            "yield_std": float(self.yield_std),
            "features_used": list(self.features_used),
        }


# ---------------------------------------------------------------------------
# Core K-NN
# ---------------------------------------------------------------------------

def _zscore(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns; returns ``(z_matrix, means, stds)``. Constant columns
    yield zero-z so they contribute nothing to the distance."""
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (matrix - mean) / std, mean, std


def analog_cone(
    geoid: str,
    target_year: int,
    as_of: str | _date,
    history_years: Sequence[int],
    weather: pd.DataFrame,
    yields: pd.DataFrame,
    k: int = DEFAULT_K,
    season_start_doy: int = DEFAULT_SEASON_START_DOY,
) -> AnalogResult:
    """K-NN analog-year cone for one (geoid, target_year, as_of).

    Args:
        geoid: 5-digit county FIPS.
        target_year: year whose forecast we're framing (e.g. 2025).
        as_of: cutoff date — analogy uses season-to-date features through this.
        history_years: candidate analog years (e.g. ``range(2008, 2025)``).
            ``target_year`` is automatically excluded if present.
        weather: merged daily weather frame (must include ``target_year``
            and every ``history_years`` entry).
        yields: NASS frame with ``geoid, year, nass_value`` columns.
        k: number of analogs to return.
        season_start_doy: growing-season start DOY (default Apr 1 / 91).

    Returns:
        :class:`AnalogResult` with the K nearest historical years' yields and
        the empirical 5-number summary.

    Raises:
        ValueError: if there isn't enough history to find ``k`` analogs with
            both a non-NaN signature and an observed NASS final.
    """
    raw_hist = sorted({int(y) for y in history_years})
    bad_history = [y for y in raw_hist if y > MAX_TRAIN_YEAR]
    if bad_history:
        raise ValueError(
            f"[2025-leak-guard] history_years includes "
            f"{bad_history} which exceed MAX_TRAIN_YEAR={MAX_TRAIN_YEAR}"
        )
    history_years = [y for y in raw_hist if y != int(target_year)]
    if len(history_years) < k:
        raise ValueError(
            f"need at least k={k} candidate history years, got "
            f"{len(history_years)}"
        )

    # 1. Build signature for the target year and every history year.
    target_sig = season_to_date_signature(
        weather, geoid=geoid, year=target_year, as_of=as_of,
        season_start_doy=season_start_doy,
    )
    history_sigs: dict[int, dict[str, float]] = {}
    for y in history_years:
        sig = season_to_date_signature(
            weather, geoid=geoid, year=y, as_of=as_of,
            season_start_doy=season_start_doy,
        )
        history_sigs[y] = sig

    # 2. Pick the feature subset that's non-NaN across target + all history
    #    years (so distance is comparable). Falls back to whatever is non-NaN
    #    in target + ≥3 history years if the strict intersect is empty.
    feature_present_strict = []
    for f in SIGNATURE_BASE_FEATURES:
        ok = (
            np.isfinite(target_sig.get(f, np.nan))
            and all(np.isfinite(history_sigs[y].get(f, np.nan)) for y in history_years)
        )
        if ok:
            feature_present_strict.append(f)

    if feature_present_strict:
        features = feature_present_strict
        candidate_years = list(history_years)
    else:
        # Loose fallback: pick features non-NaN in target + at least 3 history
        # years; drop history years missing any of those.
        loose_features: list[str] = []
        for f in SIGNATURE_BASE_FEATURES:
            if not np.isfinite(target_sig.get(f, np.nan)):
                continue
            present = sum(
                1 for y in history_years
                if np.isfinite(history_sigs[y].get(f, np.nan))
            )
            if present >= max(k, 3):
                loose_features.append(f)
        if not loose_features:
            raise ValueError(
                f"no comparable features for geoid={geoid} target_year={target_year} "
                f"as_of={as_of}; check weather coverage"
            )
        features = loose_features
        candidate_years = [
            y for y in history_years
            if all(np.isfinite(history_sigs[y].get(f, np.nan)) for f in features)
        ]
        if len(candidate_years) < k:
            raise ValueError(
                f"after filtering for shared features {features}, only "
                f"{len(candidate_years)} history years remain (need k={k})"
            )

    # 3. Filter further by yields available.
    yield_lookup: dict[int, float] = {}
    if not yields.empty:
        sub = yields[yields["geoid"].astype(str) == str(geoid)]
        for _, row in sub.iterrows():
            try:
                yield_lookup[int(row["year"])] = float(row["nass_value"])
            except (KeyError, ValueError):
                continue
    candidate_years = [y for y in candidate_years if y in yield_lookup]
    if len(candidate_years) < k:
        raise ValueError(
            f"after filtering for available NASS yields, only "
            f"{len(candidate_years)} history years remain "
            f"(need k={k}) for geoid={geoid}"
        )

    # 4. Z-score and compute distances.
    target_vec = np.asarray([target_sig[f] for f in features], dtype="float64")
    history_matrix = np.asarray(
        [[history_sigs[y][f] for f in features] for y in candidate_years],
        dtype="float64",
    )
    full = np.vstack([target_vec[None, :], history_matrix])
    z_full, _, _ = _zscore(full)
    z_target = z_full[0]
    z_history = z_full[1:]
    dists = np.sqrt(np.sum((z_history - z_target[None, :]) ** 2, axis=1))

    order = np.argsort(dists)[:k]
    chosen_years = [int(candidate_years[i]) for i in order]
    chosen_dists = [float(dists[i]) for i in order]
    chosen_yields = [float(yield_lookup[y]) for y in chosen_years]

    arr = np.asarray(chosen_yields, dtype="float64")
    return AnalogResult(
        geoid=str(geoid),
        target_year=int(target_year),
        as_of=str(_parse_as_of(as_of).isoformat()),
        k=int(k),
        analog_years=chosen_years,
        analog_distances=chosen_dists,
        analog_yields=chosen_yields,
        yield_min=float(np.min(arr)),
        yield_p25=float(np.percentile(arr, 25)),
        yield_p50=float(np.percentile(arr, 50)),
        yield_p75=float(np.percentile(arr, 75)),
        yield_max=float(np.max(arr)),
        yield_mean=float(np.mean(arr)),
        yield_std=float(np.std(arr, ddof=0)),
        features_used=list(features),
    )


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _flatten_to_row(r: AnalogResult) -> dict:
    """Flatten an AnalogResult into a parquet-friendly row."""
    base = r.to_dict()
    base["analog_years"] = ",".join(str(y) for y in r.analog_years)
    base["analog_distances"] = ",".join(f"{d:.4f}" for d in r.analog_distances)
    base["analog_yields"] = ",".join(f"{y:.2f}" for y in r.analog_yields)
    base["features_used"] = ",".join(r.features_used)
    return base


def analog_cones_for_counties(
    counties: pd.DataFrame,
    target_year: int,
    as_of: str | _date,
    history_years: Sequence[int],
    *,
    k: int = DEFAULT_K,
    weather: pd.DataFrame | None = None,
    yields: pd.DataFrame | None = None,
    include_smap: bool = True,
    include_sentinel: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Compute analog cones for a whole county catalog.

    Pulls weather + NASS once, then iterates per-county. ``weather`` and
    ``yields`` may be passed in if the caller already has them in memory
    (the forecast pipeline does); otherwise they're fetched via the cached
    engine.weather / engine.nass APIs.

    Returns one row per county with the analog cone summary plus the K
    analog years / distances / yields as comma-joined strings.
    """
    from engine.nass import fetch_counties_nass_yields
    from engine.weather import fetch_counties_weather

    raw_history = sorted({int(y) for y in history_years})
    bad_history = [y for y in raw_history if y > MAX_TRAIN_YEAR]
    if bad_history:
        raise ValueError(
            f"[2025-leak-guard] history_years includes "
            f"{bad_history} which exceed MAX_TRAIN_YEAR={MAX_TRAIN_YEAR}"
        )
    history_years = [y for y in raw_history if y != int(target_year)]
    if not history_years:
        raise ValueError("history_years is empty after excluding target_year")

    banner(
        f"ANALOG CONE  target_year={target_year}  as_of={as_of}  "
        f"k={k}  history={history_years[0]}-{history_years[-1]}  "
        f"counties={len(counties)}",
        logger=logger,
    )

    if weather is None:
        all_years = sorted(set([int(target_year), *history_years]))
        weather = fetch_counties_weather(
            counties,
            start_year=min(all_years),
            end_year=max(all_years),
            include_smap=include_smap,
            include_sentinel=include_sentinel,
            refresh=refresh,
        )
    if yields is None:
        yields = fetch_counties_nass_yields(
            counties,
            start_year=min(history_years),
            end_year=max(history_years),
            refresh=refresh,
        )

    rows: list[dict] = []
    sc = StepCounter(logger, total=len(counties), unit="counties", every=25,
                     prefix="analogs")
    for _, county in counties.iterrows():
        geoid = str(county["geoid"])
        try:
            res = analog_cone(
                geoid=geoid,
                target_year=int(target_year),
                as_of=as_of,
                history_years=history_years,
                weather=weather,
                yields=yields,
                k=k,
            )
        except ValueError as exc:
            logger.warning("analog_cone skipped for geoid=%s: %s", geoid, exc)
            sc.tick(extra=f"{geoid} (skipped)")
            continue
        rows.append(_flatten_to_row(res))
        logger.debug(
            "analog: geoid=%s target=%d as_of=%s analogs=%s yields=%s "
            "P50=%.1f cone[min,max]=[%.1f,%.1f]",
            geoid, int(target_year), res.as_of, res.analog_years,
            [round(y, 1) for y in res.analog_yields],
            res.yield_p50, res.yield_min, res.yield_max,
        )
        sc.tick()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI: hack26-analogs --inspect
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or batch-compute analog-year cones."
    )
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--inspect", action="store_true",
                     help="Single-county inspect mode. Requires --geoid.")
    sub.add_argument("--batch", action="store_true",
                     help="Batch-mode over a state catalog. Writes parquet/CSV.")

    parser.add_argument("--geoid", type=str, default=None,
                        help="(--inspect) 5-digit county FIPS, e.g. 19169.")
    parser.add_argument("--states", nargs="+", default=None, metavar="STATE",
                        help="(--batch) subset of states.")
    parser.add_argument("--target-year", type=int, default=2025,
                        help="Year to find analogs for (default 2025).")
    parser.add_argument("--as-of", type=str, default="2025-08-01",
                        help="Cutoff date (ISO YYYY-MM-DD).")
    parser.add_argument("--history-start", type=int, default=MIN_TRAIN_YEAR,
                        help=f"Earliest history year (default {MIN_TRAIN_YEAR}).")
    parser.add_argument("--history-end", type=int, default=MAX_TRAIN_YEAR,
                        help=f"Latest history year (default {MAX_TRAIN_YEAR}; "
                             f"capped to {MAX_TRAIN_YEAR} by 2025-leak guard).")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--include-sentinel", action="store_true")
    parser.add_argument("--no-smap", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", type=Path, default=None,
                        help="(--batch) output path (.parquet or .csv).")
    add_cli_logging_args(parser)
    args = parser.parse_args(argv)

    log_path = apply_cli_logging_args(args, tag="analogs")
    log_environment(logger)
    logger.info("rotated log: %s", log_path)

    history_years = list(range(args.history_start, args.history_end + 1))

    if args.inspect:
        if not args.geoid:
            parser.error("--inspect requires --geoid")
        from engine.counties import load_counties
        from engine.nass import fetch_counties_nass_yields
        from engine.weather import fetch_counties_weather

        all_counties = load_counties()
        county_row = all_counties[all_counties["geoid"] == args.geoid]
        if county_row.empty:
            parser.error(f"--geoid {args.geoid!r} not in catalog "
                         "(must be one of the 5 target states' FIPS)")
        years = sorted({int(args.target_year), *history_years})
        weather = fetch_counties_weather(
            county_row,
            start_year=min(years), end_year=max(years),
            include_smap=not args.no_smap,
            include_sentinel=args.include_sentinel,
            refresh=args.refresh,
        )
        yields = fetch_counties_nass_yields(
            county_row,
            start_year=args.history_start, end_year=args.history_end,
            refresh=args.refresh,
        )
        try:
            res = analog_cone(
                geoid=args.geoid, target_year=args.target_year,
                as_of=args.as_of, history_years=history_years,
                weather=weather, yields=yields, k=args.k,
            )
        except ValueError as exc:
            logger.error(str(exc))
            return 2
        import json as _json
        print(_json.dumps(res.to_dict(), indent=2))
        return 0

    # batch mode
    from engine.counties import load_counties
    counties = load_counties(states=args.states)
    df = analog_cones_for_counties(
        counties=counties,
        target_year=args.target_year,
        as_of=args.as_of,
        history_years=history_years,
        k=args.k,
        include_smap=not args.no_smap,
        include_sentinel=args.include_sentinel,
        refresh=args.refresh,
    )
    logger.info("analog batch result: rows=%d", len(df))
    with pd.option_context("display.max_rows", 12, "display.width", 200):
        print(df, file=sys.stderr)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        suf = args.out.suffix.lower()
        if suf == ".parquet":
            df.to_parquet(args.out, index=False)
        elif suf == ".csv":
            df.to_csv(args.out, index=False)
        else:
            parser.error(f"--out must be .parquet or .csv (got {suf})")
        logger.info("wrote %s (%d rows)", args.out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
