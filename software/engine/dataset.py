"""Assemble Darts ``TimeSeries`` bundles from the existing engine sources.

This is the bridge between the SPEC §2 ``fetch(...)`` data sources and the
TFT model defined in :mod:`engine.model`. One *(geoid, year)* tuple becomes
one Darts ``TimeSeries`` of length ``season_end_doy - season_start_doy + 1``
(244 days for Apr 1 → Nov 30 by default).

Public surface:

    TrainingBundle         dataclass — target/past/future series + statics
    build_training_dataset(states, start_year, end_year, ...) -> TrainingBundle
    build_inference_dataset(states, target_year, ...)         -> TrainingBundle
    save_training_bundle / load_training_bundle — pickle the bundle for
        ``hack26-train`` to reuse (avoids re-pulling weather/CDL for every run)
    default_last_training_bundle_path — canonical path for ``--save-last-bundle``
    _main(argv)            CLI:  python -m engine.dataset --dump-stats ...

Hard rule (enforced in code): ``end_year`` may not exceed
:data:`MAX_TRAIN_YEAR` (= 2024). 2025 is the deliverable forecast year and
is the only true out-of-sample benchmark — it MUST NOT be used for training,
validation, or feature engineering. Use :func:`build_inference_dataset`
(no labels required) for 2025.

Feature design:

- **Target series** — yield (bu/ac) broadcast across the whole series. At
  training, the broadcast value is the realized NASS final. At inference
  (no label), the broadcast value is the per-county historical mean yield
  (in-distribution numeric prior so the encoder isn't fed zeros).
- **Past observed covariates** — daily weather features from
  :mod:`engine.weather` (temperature, precip, soil moisture, NDVI/NDWI when
  available, GDD + cumulative GDD, 7d/30d rollups). Imputed forward-then-mean
  per (geoid, year) to give Darts a clean dense series.
- **Future known covariates** — purely calendar (DOY sin/cos, week sin/cos,
  month, days-until-end-of-season). Live weather forecasts (GEFS) are a v1.1
  add and are an explicit non-goal here.
- **Static covariates** — per-(geoid, year): CDL corn/soy fractions, county
  centroid + log-area, plus per-county *historical-mean yield* computed
  exclusively from years strictly before ``year`` (no leakage).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

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

if TYPE_CHECKING:  # avoid pulling Darts into module import time
    from darts import TimeSeries  # noqa: F401

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public constants — the 2025 leak guard lives here
# ---------------------------------------------------------------------------

#: Hard ceiling for any training/validation/test year. 2025 is the deliverable
#: forecast and the only out-of-sample benchmark — it is never allowed to
#: appear as a label or in feature-engineering aggregations.
MAX_TRAIN_YEAR = 2024

#: Earliest year with both NASS labels and acceptable POWER coverage.
MIN_TRAIN_YEAR = 2008

#: Default growing-season window. April 1 (DOY 91) → November 30 (DOY 334).
DEFAULT_SEASON_START_DOY = 91
DEFAULT_SEASON_END_DOY = 334

#: State FIPS → integer index for the categorical embedding in TFT. Order is
#: stable (sorted by FIPS) so a model trained today loads a model trained
#: last week without an embedding-row mismatch.
STATE_FIPS_INDEX: dict[str, int] = {
    "08": 0,  # Colorado
    "19": 1,  # Iowa
    "29": 2,  # Missouri
    "31": 3,  # Nebraska
    "55": 4,  # Wisconsin
}

#: Past-observed covariate columns we'll use if present in the merged weather
#: frame. Anything missing is silently dropped at series-build time so the
#: bundle stays robust when Sentinel/SMAP isn't available for older years.
PAST_COVARIATE_BASE_COLS: tuple[str, ...] = (
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "GWETROOT",
    "GWETTOP",
    "GWETPROF",
    "SMAP_surface_sm_m3m3",
    "NDVI",
    "NDWI",
    "GDD",
    "GDD_cumulative",
)

#: Rolling-feature suffixes from :mod:`engine.weather.features`.
PAST_COVARIATE_ROLLING_SUFFIXES: tuple[str, ...] = ("_7d_avg", "_30d_avg")

#: Calendar features published as known-future covariates.
FUTURE_COVARIATE_COLS: tuple[str, ...] = (
    "doy_sin",
    "doy_cos",
    "week_sin",
    "week_cos",
    "month",
    "days_until_end_of_season",
)

#: One-hot static columns for state. Treating state as a learned embedding
#: via Darts' ``categorical_static_covariates`` is cleaner in theory, but
#: the API has shifted across versions and a 5-way one-hot is functionally
#: equivalent at this cardinality with zero compatibility risk.
STATE_ONEHOT_COLS: tuple[str, ...] = tuple(
    f"state_{fips}" for fips in sorted(STATE_FIPS_INDEX.keys())
)

#: Static covariate columns. Order matters — Darts keys static covariates by
#: name, but keeping a stable order makes serialized bundles diffable.
STATIC_COVARIATE_COLS: tuple[str, ...] = (
    *STATE_ONEHOT_COLS,
    "corn_pct_of_county",
    "corn_pct_of_cropland",
    "soybean_pct_of_cropland",
    "log_corn_area_m2",
    "log_land_area_m2",
    "centroid_lat",
    "centroid_lon",
    "historical_mean_yield_bu_acre",
)


# ---------------------------------------------------------------------------
# TrainingBundle
# ---------------------------------------------------------------------------

@dataclass
class TrainingBundle:
    """Container returned by :func:`build_training_dataset`.

    Each list is parallel — ``target_series[i]``, ``past_covariates[i]``,
    ``future_covariates[i]``, ``series_index.iloc[i]``, and
    ``static_covariates.iloc[i]`` all describe the same *(geoid, year)*.
    """

    #: One target series per (geoid, year). Length = season days.
    target_series: list = field(default_factory=list)

    #: Daily weather features per (geoid, year), same length as target.
    past_covariates: list = field(default_factory=list)

    #: Calendar features per (geoid, year), same length as target.
    future_covariates: list = field(default_factory=list)

    #: One row per series with `STATIC_COVARIATE_COLS`.
    static_covariates: pd.DataFrame = field(default_factory=pd.DataFrame)

    #: One row per series — `geoid, year, state_fips, label, label_present`.
    series_index: pd.DataFrame = field(default_factory=pd.DataFrame)

    #: List of past-covariate column names actually used (after dropping
    #: those entirely missing in the source frame).
    past_covariate_cols: list[str] = field(default_factory=list)

    #: List of static covariate column names (always == STATIC_COVARIATE_COLS,
    #: surfaced for symmetry and so model code doesn't need to import the
    #: constant).
    static_covariate_cols: list[str] = field(default_factory=list)

    @property
    def n_series(self) -> int:
        return len(self.target_series)

    def filter_by_year(self, years: Iterable[int]) -> "TrainingBundle":
        """Return a new bundle with only series whose ``year`` is in ``years``.

        Used by the model layer to split train / val / test along chronological
        boundaries. Lists are filtered in lockstep with ``series_index``.
        """
        wanted = {int(y) for y in years}
        keep_mask = self.series_index["year"].astype(int).isin(wanted).to_numpy()
        keep_idx = np.flatnonzero(keep_mask).tolist()
        return TrainingBundle(
            target_series=[self.target_series[i] for i in keep_idx],
            past_covariates=[self.past_covariates[i] for i in keep_idx],
            future_covariates=[self.future_covariates[i] for i in keep_idx],
            static_covariates=self.static_covariates.iloc[keep_idx].reset_index(drop=True),
            series_index=self.series_index.iloc[keep_idx].reset_index(drop=True),
            past_covariate_cols=list(self.past_covariate_cols),
            static_covariate_cols=list(self.static_covariate_cols),
        )


# ---------------------------------------------------------------------------
# Training bundle on-disk cache (for hack26-dataset → hack26-train)
# ---------------------------------------------------------------------------

BUNDLE_META_VERSION: int = 2  # bumped: enforce float32 covariates / statics
LAST_BUNDLE_PKL: str = "last_training_bundle.pkl"
LAST_BUNDLE_META: str = "last_training_bundle.meta.json"


def _bundles_dir() -> Path:
    """``<data_root>/derived/bundles/`` — same data root as weather/NASS."""
    from engine.nass._cache import data_root

    d = data_root() / "derived" / "bundles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_last_training_bundle_path() -> Path:
    """Default pickle path written by ``hack26-dataset --save-last-bundle``."""
    return _bundles_dir() / LAST_BUNDLE_PKL


def default_last_training_bundle_meta_path() -> Path:
    return _bundles_dir() / LAST_BUNDLE_META


def _meta_path_for_pickle(pkl: Path) -> Path:
    """Sidecar JSON next to the pickle (e.g. ``foo.pkl`` → ``foo.meta.json``)."""
    return pkl.with_name(pkl.stem + ".meta.json")


def save_training_bundle(
    bundle: TrainingBundle,
    pkl_path: Path,
    *,
    states_fips: Sequence[str],
    start_year: int,
    end_year: int,
    include_sentinel: bool,
    include_smap: bool,
    season_start_doy: int = DEFAULT_SEASON_START_DOY,
    season_end_doy: int = DEFAULT_SEASON_END_DOY,
    require_label: bool = True,
    min_coverage_frac: float = 0.95,
) -> Path:
    """Pickle a :class:`TrainingBundle` + write JSON metadata for train-time checks.

    **Security:** Only :func:`load_training_bundle` pickles you created; never
    load untrusted files.

    Returns:
        ``pkl_path`` (written).
    """
    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "bundle_meta_version": BUNDLE_META_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "states_fips": sorted({str(f).zfill(2) for f in states_fips}),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "include_sentinel": bool(include_sentinel),
        "include_smap": bool(include_smap),
        "season_start_doy": int(season_start_doy),
        "season_end_doy": int(season_end_doy),
        "require_label": bool(require_label),
        "min_coverage_frac": float(min_coverage_frac),
    }
    meta_path = _meta_path_for_pickle(pkl_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    with pkl_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path


def load_training_bundle(pkl_path: Path) -> TrainingBundle:
    """Load a bundle saved with :func:`save_training_bundle`."""
    with Path(pkl_path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TrainingBundle):
        raise TypeError(f"expected TrainingBundle, got {type(obj).__name__}")
    return obj


def load_training_bundle_meta(pkl_path: Path) -> dict[str, Any] | None:
    """Return metadata dict if the sidecar exists, else ``None``."""
    mp = _meta_path_for_pickle(Path(pkl_path))
    if not mp.is_file():
        return None
    with mp.open(encoding="utf-8") as f:
        return json.load(f)


def training_bundle_fits_train_request(
    bundle: TrainingBundle,
    meta: dict[str, Any] | None,
    *,
    states_fips: list[str],
    required_years: set[int],
    include_sentinel: bool,
    include_smap: bool,
    season_start_doy: int = DEFAULT_SEASON_START_DOY,
    season_end_doy: int = DEFAULT_SEASON_END_DOY,
) -> tuple[bool, str]:
    """Return (True, \"\") if a cached bundle matches the training CLI request.

    ``required_years`` is typically ``train_years ∪ {val, test}`` clipped to
    :data:`MAX_TRAIN_YEAR`. Every required year must appear at least once in
    ``bundle.series_index['year']``. State lists (sorted FIPS) must match.
    """
    want_states = sorted({str(f).zfill(2) for f in states_fips})
    if meta is not None:
        if int(meta.get("bundle_meta_version", 0)) != BUNDLE_META_VERSION:
            return False, f"meta version mismatch (got {meta.get('bundle_meta_version')!r})"
        meta_states = list(meta.get("states_fips") or [])
        if sorted(str(x).zfill(2) for x in meta_states) != want_states:
            return (
                False,
                f"states mismatch: cache has {meta_states!r}, train needs {want_states!r}",
            )
        if bool(meta.get("include_sentinel")) != include_sentinel:
            return False, "include_sentinel mismatch"
        if bool(meta.get("include_smap")) != include_smap:
            return False, "include_smap mismatch"
        if int(meta.get("season_start_doy", -1)) != int(season_start_doy):
            return False, "season_start_doy mismatch"
        if int(meta.get("season_end_doy", -1)) != int(season_end_doy):
            return False, "season_end_doy mismatch"
    else:
        g = bundle.series_index
        if g.empty or "state_fips" not in g.columns:
            return False, "no meta.json and series_index is unusable"
        have = sorted(
            {str(s).zfill(2) for s in g["state_fips"].astype(str).unique().tolist()}
        )
        if have != want_states:
            return False, f"states in bundle {have!r} != train {want_states!r}"

    if bundle.series_index.empty:
        return False, "empty bundle"

    have_years = set(bundle.series_index["year"].astype(int).tolist())
    miss = set(required_years) - have_years
    if miss:
        return (
            False,
            f"cached bundle missing year(s): {sorted(miss)} (have {min(have_years)}..{max(have_years)})",
        )
    if meta is not None:
        if int(meta["start_year"]) > min(required_years):
            return (
                False,
                f"cache start_year {meta['start_year']} > min required {min(required_years)}",
            )
        if int(meta["end_year"]) < max(required_years):
            return (
                False,
                f"cache end_year {meta['end_year']} < max required {max(required_years)}",
            )
    return True, ""


# ---------------------------------------------------------------------------
# Year-range validation
# ---------------------------------------------------------------------------

def _assert_year_in_training_range(year: int, role: str) -> None:
    """Raise if ``year`` is outside the training window, with a diagnostic
    that mentions the 2025 leak guard explicitly.
    """
    y = int(year)
    if y > MAX_TRAIN_YEAR:
        raise ValueError(
            f"{role} year {y} is past the 2025-strict-holdout cutoff "
            f"(MAX_TRAIN_YEAR={MAX_TRAIN_YEAR}). "
            f"2025 is the deliverable forecast year and may not be used "
            f"for training, validation, or feature engineering. "
            f"Use build_inference_dataset() for 2025."
        )
    if y < MIN_TRAIN_YEAR:
        raise ValueError(
            f"{role} year {y} predates available NASS county finals "
            f"(MIN_TRAIN_YEAR={MIN_TRAIN_YEAR})."
        )


def _validate_train_year_range(start_year: int, end_year: int) -> None:
    if start_year > end_year:
        raise ValueError(
            f"start_year ({start_year}) cannot be after end_year ({end_year})"
        )
    _assert_year_in_training_range(start_year, role="start")
    _assert_year_in_training_range(end_year, role="end")


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _filter_to_growing_season(
    df: pd.DataFrame,
    start_doy: int,
    end_doy: int,
) -> pd.DataFrame:
    """Slice the merged daily weather frame to growing-season days only."""
    if df.empty:
        return df
    dates = df.index.get_level_values("date")
    doy = dates.dayofyear
    keep = (doy >= start_doy) & (doy <= end_doy)
    return df[keep]


def _select_past_covariate_cols(df: pd.DataFrame) -> list[str]:
    """Return the past-covariate columns actually present in ``df``.

    Honors :data:`PAST_COVARIATE_BASE_COLS` plus their ``_7d_avg`` and
    ``_30d_avg`` rollups. Skips columns that are entirely NaN (pre-SMAP
    years, no-Sentinel runs) so the model isn't fed dead features.
    """
    candidates: list[str] = []
    for base in PAST_COVARIATE_BASE_COLS:
        if base in df.columns:
            candidates.append(base)
        for suf in PAST_COVARIATE_ROLLING_SUFFIXES:
            col = f"{base}{suf}"
            if col in df.columns:
                candidates.append(col)

    out: list[str] = []
    for col in candidates:
        nonnull = df[col].notna().sum()
        if nonnull > 0:
            out.append(col)
        else:
            logger.debug(
                "dropping past-covariate column %s (all-NaN across %d rows)",
                col, len(df),
            )
    return out


def _build_calendar_frame(
    dates: pd.DatetimeIndex, season_end_doy: int
) -> pd.DataFrame:
    """Construct the known-future calendar covariates for one series."""
    doy = dates.dayofyear.to_numpy()
    week = dates.isocalendar().week.to_numpy().astype(np.int32)
    month = dates.month.to_numpy().astype(np.int32)
    return pd.DataFrame(
        {
            "doy_sin": np.sin(2 * np.pi * doy / 366.0).astype(np.float32),
            "doy_cos": np.cos(2 * np.pi * doy / 366.0).astype(np.float32),
            "week_sin": np.sin(2 * np.pi * week / 53.0).astype(np.float32),
            "week_cos": np.cos(2 * np.pi * week / 53.0).astype(np.float32),
            "month": month.astype(np.float32),
            "days_until_end_of_season":
                (season_end_doy - doy).astype(np.float32),
        },
        index=dates,
    )


def _impute_past_block(
    block: pd.DataFrame, cols: Sequence[str]
) -> pd.DataFrame:
    """Forward-fill then back-fill then mean-fill each column. Robust to a
    column being entirely missing for one (geoid, year) (e.g. pre-SMAP years).
    """
    out = block[list(cols)].copy()
    out = out.ffill().bfill()
    if out.isna().any().any():
        # Fall back to per-column mean (training-frame-wide) — last resort.
        means = out.mean(numeric_only=True)
        out = out.fillna(means).fillna(0.0)
    return out.astype(np.float32)


def _historical_mean_yields(
    yields: pd.DataFrame, target_year: int
) -> pd.Series:
    """Per-county mean yield over **strictly prior** years.

    Returns a Series indexed by ``geoid``. Counties that have no prior years
    of observed yield get the global mean over prior years. Ensures the
    static covariate `historical_mean_yield_bu_acre` is leak-free.
    """
    prior = yields[yields["year"].astype(int) < int(target_year)]
    per_county = prior.groupby("geoid")["nass_value"].mean()
    if per_county.empty:
        return pd.Series(dtype="float64")
    global_mean = float(prior["nass_value"].mean())
    return per_county.reindex(prior["geoid"].unique(), fill_value=global_mean)


# ---------------------------------------------------------------------------
# Source orchestration
# ---------------------------------------------------------------------------

def _load_sources(
    states: Sequence[str] | None,
    start_year: int,
    end_year: int,
    include_sentinel: bool,
    include_smap: bool,
    refresh: bool,
    allow_download: bool = False,
    max_fetch_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pull counties + weather + cdl + nass through the existing engine APIs.

    Returns ``(counties, weather_daily, cdl_annual, nass_yields)``. Every
    sub-source uses its own parquet cache; the first cold pull is the slow
    one (POWER for 17 years × 443 counties). Subsequent calls are cache reads.

    ``max_fetch_workers`` controls parallel I/O: NASA POWER/SMAP/Sentinel per
    county, CDL per year, and NASS per state (``1`` disables — legacy
    sequential behavior for weather layers that use ``sleep_between``).
    """
    from engine.cdl import fetch_counties_cdl
    from engine.counties import load_counties
    from engine.nass import fetch_counties_nass_yields
    from engine.weather import fetch_counties_weather

    banner("STEP 1/5  Loading county catalog", logger=logger)
    counties = load_counties(states=states)
    logger.info(
        "counties loaded: n=%d  states=%s",
        len(counties), sorted(counties["state_fips"].unique().tolist()),
    )

    banner(
        f"STEP 2/5  Pulling weather (POWER+SMAP{'+Sentinel' if include_sentinel else ''}) "
        f"for {start_year}-{end_year}",
        logger=logger,
    )
    w = max(1, int(max_fetch_workers))
    weather = fetch_counties_weather(
        counties,
        start_year=start_year,
        end_year=end_year,
        include_smap=include_smap,
        include_sentinel=include_sentinel,
        refresh=refresh,
        max_workers=w,
    )
    logger.info(
        "weather frame: rows=%d  cols=%d  index=%s",
        len(weather), len(weather.columns), list(weather.index.names),
    )

    banner(
        f"STEP 3/5  Pulling CDL annual snapshots for {start_year}-{end_year}",
        logger=logger,
    )
    years_list = list(range(start_year, end_year + 1))
    n_years = len(years_list)
    cdl_workers = min(w, n_years) if n_years else 1

    def _cdl_one_year(yr: int) -> tuple[int, pd.DataFrame | None, str | None]:
        res = 30
        try:
            df = fetch_counties_cdl(
                counties, year=yr, resolution=res,
                refresh=refresh, allow_download=allow_download,
            )
            return (yr, df.assign(year=yr), None)
        except Exception as exc:  # noqa: BLE001
            return (yr, None, str(exc))

    cdl_pieces: list[pd.DataFrame] = []
    if cdl_workers <= 1 or n_years <= 1:
        sc = StepCounter(logger, total=n_years, unit="years",
                         every=1, prefix="cdl-years")
        for yr in years_list:
            out_yr, df, err = _cdl_one_year(yr)
            if err is not None:
                logger.warning(
                    "CDL year %d unavailable (%s); proceeding without it. "
                    "Static covariates for that year will fall back to the "
                    "nearest available CDL year.",
                    out_yr, err,
                )
            elif df is not None:
                cdl_pieces.append(df)
                sc.tick(extra=f"year={out_yr} rows={len(df)}")
    else:
        logger.info(
            "CDL years: parallel fetch (max_workers=%d) for %d years",
            cdl_workers, n_years,
        )
        sc = StepCounter(logger, total=n_years, unit="years",
                         every=1, prefix="cdl-years")
        with ThreadPoolExecutor(max_workers=cdl_workers) as ex:
            futures = {ex.submit(_cdl_one_year, yr): yr for yr in years_list}
            for fut in as_completed(futures):
                out_yr, df, err = fut.result()
                if err is not None:
                    logger.warning(
                        "CDL year %d unavailable (%s); proceeding without it. "
                        "Static covariates for that year will fall back to the "
                        "nearest available CDL year.",
                        out_yr, err,
                    )
                elif df is not None:
                    cdl_pieces.append(df)
                    sc.tick(extra=f"year={out_yr} rows={len(df)}")
        # Restore chronological order for logging / sanity
        cdl_pieces.sort(key=lambda d: int(d["year"].iloc[0]))

    cdl = pd.concat(cdl_pieces, ignore_index=True) if cdl_pieces else pd.DataFrame()
    logger.info(
        "cdl frame: rows=%d  years=%d  geoids=%d",
        len(cdl),
        cdl["year"].nunique() if not cdl.empty else 0,
        cdl["geoid"].nunique() if not cdl.empty else 0,
    )

    banner(
        f"STEP 4/5  Pulling NASS county yields (labels) for {start_year}-{end_year}",
        logger=logger,
    )
    nass = fetch_counties_nass_yields(
        counties,
        start_year=start_year,
        end_year=end_year,
        refresh=refresh,
        max_workers=w,
    )
    # Hard guard: drop any 2025 row in case a refreshed NASS pull has it (NASS
    # doesn't publish 2025 finals yet, but the guard keeps us honest).
    if (nass["year"].astype(int) == 2025).any():
        n_dropped = int((nass["year"].astype(int) == 2025).sum())
        logger.warning(
            "[2025-leak-guard] dropping %d NASS rows tagged year==2025 "
            "before label assignment", n_dropped,
        )
        nass = nass[nass["year"].astype(int) != 2025].copy()
    logger.info(
        "nass frame: rows=%d  geoids=%d  years=%s..%s",
        len(nass),
        nass["geoid"].nunique() if not nass.empty else 0,
        int(nass["year"].min()) if not nass.empty else "-",
        int(nass["year"].max()) if not nass.empty else "-",
    )

    return counties, weather, cdl, nass


def _build_static_row(
    geoid: str,
    year: int,
    county_row: pd.Series,
    cdl_row: pd.Series | None,
    historical_mean: float,
) -> dict[str, float]:
    """Static covariates for one (geoid, year)."""
    if cdl_row is None:
        # Fallback when the CDL year is missing: zeros for crop fractions.
        # Logged at warning above; downstream model will see these as a known
        # "no CDL signal" row.
        corn_pct_county = 0.0
        corn_pct_cropland = 0.0
        soy_pct_cropland = 0.0
        log_corn_area = 0.0
    else:
        corn_pct_county = float(cdl_row.get("corn_pct_of_county") or 0.0)
        corn_pct_cropland = float(cdl_row.get("corn_pct_of_cropland") or 0.0)
        soy_px = float(cdl_row.get("soybean_pixels") or 0.0)
        cropland_px = float(cdl_row.get("cropland_pixels") or 0.0)
        soy_pct_cropland = soy_px / cropland_px if cropland_px > 0 else 0.0
        corn_area = float(cdl_row.get("corn_area_m2") or 0.0)
        log_corn_area = float(np.log1p(corn_area))

    land_area = float(county_row.get("land_area_m2") or 0.0)

    state_fips = geoid[:2]
    row: dict[str, float] = {col: 0.0 for col in STATE_ONEHOT_COLS}
    onehot_col = f"state_{state_fips}"
    if onehot_col in row:
        row[onehot_col] = 1.0
    row.update({
        "corn_pct_of_county": corn_pct_county,
        "corn_pct_of_cropland": corn_pct_cropland,
        "soybean_pct_of_cropland": soy_pct_cropland,
        "log_corn_area_m2": log_corn_area,
        "log_land_area_m2": float(np.log1p(land_area)),
        "centroid_lat": float(county_row.get("centroid_lat") or 0.0),
        "centroid_lon": float(county_row.get("centroid_lon") or 0.0),
        "historical_mean_yield_bu_acre": float(historical_mean),
    })
    return row


def _build_series_for_county_year(
    geoid: str,
    year: int,
    weather_block: pd.DataFrame,
    past_cols: Sequence[str],
    season_dates: pd.DatetimeIndex,
    season_end_doy: int,
    label: float,
    static_row: dict[str, float],
):
    """Construct the (target, past, future) Darts TimeSeries triple.

    Reindexes ``weather_block`` onto the canonical season ``DatetimeIndex``
    (so all series have identical length and timestamps) and imputes any
    holes per-column.

    Returns a 3-tuple of ``TimeSeries`` objects. Static covariates are
    attached to the **target** series (Darts convention).
    """
    from darts import TimeSeries

    # Canonical season index for this (geoid, year). All three series share
    # exactly this index so Darts' shape checks pass.
    weather_block = weather_block.reset_index(level="geoid", drop=True)
    weather_block = weather_block.reindex(season_dates)
    past_df = _impute_past_block(weather_block, past_cols)
    future_df = _build_calendar_frame(season_dates, season_end_doy=season_end_doy)

    # Darts' TFT runs in float32; mixed-precision inputs raise
    # "expected m1 and m2 to have the same dtype" in the prescaler Linear layer.
    static_df = pd.DataFrame([static_row], index=[0]).astype(np.float32)

    target_values = np.full(
        (len(season_dates), 1), float(label), dtype=np.float32
    )
    target_df = pd.DataFrame(
        target_values,
        index=season_dates,
        columns=["yield_bu_acre"],
    )

    past_df = past_df.astype(np.float32)
    future_df = future_df.astype(np.float32)

    target_ts = TimeSeries.from_dataframe(
        target_df,
        freq="D",
        static_covariates=static_df,
    )
    past_ts = TimeSeries.from_dataframe(past_df, freq="D")
    future_ts = TimeSeries.from_dataframe(future_df, freq="D")

    # Sanity: Darts is unforgiving about silent NaN in inputs.
    if not np.all(np.isfinite(past_ts.values())):
        raise ValueError(
            f"past covariates contain non-finite values for "
            f"geoid={geoid} year={year}"
        )
    if not np.all(np.isfinite(future_ts.values())):
        raise ValueError(
            f"future covariates contain non-finite values for "
            f"geoid={geoid} year={year}"
        )

    return target_ts, past_ts, future_ts


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_training_dataset(
    states: Sequence[str] | None = None,
    start_year: int = MIN_TRAIN_YEAR,
    end_year: int = MAX_TRAIN_YEAR,
    season_start_doy: int = DEFAULT_SEASON_START_DOY,
    season_end_doy: int = DEFAULT_SEASON_END_DOY,
    include_sentinel: bool = False,
    include_smap: bool = True,
    refresh: bool = False,
    require_label: bool = True,
    min_coverage_frac: float = 0.95,
    allow_download: bool = False,
    max_fetch_workers: int = 4,
) -> TrainingBundle:
    """Assemble the per-(geoid, year) Darts bundle for **training** roles.

    Args:
        states: subset (state names or 2-digit FIPS); ``None`` = all 5.
        start_year / end_year: inclusive year range. Both are clamped to
            ``[MIN_TRAIN_YEAR, MAX_TRAIN_YEAR]`` — ``end_year > 2024`` raises.
        season_start_doy / season_end_doy: growing-season window, inclusive.
        include_sentinel: pull NDVI/NDWI from Sentinel-2 (slow & needs the
            ``[sentinel]`` extra; off by default).
        include_smap: pull SMAP soil moisture (free with POWER endpoint).
        refresh: force re-download from POWER/SMAP/Sentinel/CDL/NASS.
        require_label: drop series with no NASS final yield. Set ``False`` to
            keep label-less series for an inference build (use
            :func:`build_inference_dataset` for that instead).
        min_coverage_frac: drop series whose past-covariate non-NaN coverage
            (after impute) falls below this — guards against malformed weather
            pulls.
        max_fetch_workers: parallel I/O for weather counties, CDL years, and
            NASS states (``1`` = fully sequential, legacy throttling for POWER).

    Returns:
        :class:`TrainingBundle`. ``n_series`` ≈ ``n_counties * (end - start + 1)``
        before drops; expect ~5-10% drops from missing labels in early years.
    """
    _validate_train_year_range(start_year, end_year)

    counties, weather, cdl, nass = _load_sources(
        states=states,
        start_year=start_year,
        end_year=end_year,
        include_sentinel=include_sentinel,
        include_smap=include_smap,
        refresh=refresh,
        allow_download=allow_download,
        max_fetch_workers=max_fetch_workers,
    )

    banner("STEP 5/5  Assembling Darts TimeSeries", logger=logger)

    if weather.empty:
        raise RuntimeError("weather frame is empty; cannot build training set")

    weather_season = _filter_to_growing_season(
        weather, start_doy=season_start_doy, end_doy=season_end_doy
    )
    past_cols = _select_past_covariate_cols(weather_season)
    if not past_cols:
        raise RuntimeError(
            "no usable past-covariate columns survived the all-NaN filter; "
            "check the weather pull"
        )
    logger.info("past-covariate columns (%d): %s", len(past_cols), past_cols)
    logger.info("future-covariate columns (%d): %s",
                len(FUTURE_COVARIATE_COLS), list(FUTURE_COVARIATE_COLS))
    logger.info("static-covariate columns  (%d): %s",
                len(STATIC_COVARIATE_COLS), list(STATIC_COVARIATE_COLS))

    # Pre-index the auxiliary frames for O(1) lookup inside the inner loop.
    counties_by_geoid = counties.set_index("geoid")
    cdl_by_year_geoid: dict[tuple[int, str], pd.Series] = {}
    if not cdl.empty:
        for (yr, gid), grp in cdl.groupby(["year", "geoid"]):
            cdl_by_year_geoid[(int(yr), str(gid))] = grp.iloc[0]

    nass_by_year_geoid: dict[tuple[int, str], float] = {}
    for _, row in nass.iterrows():
        nass_by_year_geoid[(int(row["year"]), str(row["geoid"]))] = float(
            row["nass_value"]
        )

    # Per-target-year historical mean yields (computed strictly from prior
    # years inside the training window — no leakage).
    hist_mean_by_year: dict[int, pd.Series] = {}
    for yr in range(start_year, end_year + 1):
        hist_mean_by_year[yr] = _historical_mean_yields(nass, target_year=yr)

    target_series: list = []
    past_series: list = []
    future_series: list = []
    static_rows: list[dict[str, float]] = []
    index_rows: list[dict] = []

    n_total = len(counties) * (end_year - start_year + 1)
    sc = StepCounter(logger, total=n_total, unit="series", every=200,
                     prefix="series-built")

    n_dropped_label = 0
    n_dropped_coverage = 0
    n_dropped_no_weather = 0

    for _, county in counties.iterrows():
        geoid = str(county["geoid"])

        # All weather rows for this county at once — orders of magnitude
        # faster than slicing inside the year loop.
        try:
            county_block = weather_season.xs(geoid, level="geoid", drop_level=False)
        except KeyError:
            sc.tick(extra=f"geoid={geoid} (no weather)")
            n_dropped_no_weather += end_year - start_year + 1
            continue

        for yr in range(start_year, end_year + 1):
            label = nass_by_year_geoid.get((yr, geoid))
            if label is None:
                if require_label:
                    n_dropped_label += 1
                    sc.tick(extra=f"geoid={geoid} year={yr} (no label)")
                    continue
                # Inference build: use the per-county historical mean as the
                # broadcast target placeholder.
                hist_mean = hist_mean_by_year.get(yr, pd.Series(dtype="float64"))
                label = float(hist_mean.get(geoid, hist_mean.mean() if len(hist_mean) else 0.0))

            season_dates = pd.date_range(
                start=pd.Timestamp(year=yr, month=1, day=1)
                + pd.Timedelta(days=season_start_doy - 1),
                end=pd.Timestamp(year=yr, month=1, day=1)
                + pd.Timedelta(days=season_end_doy - 1),
                freq="D",
            )

            # Slice that county-block to the target year.
            yr_dates = county_block.index.get_level_values("date")
            yr_mask = (yr_dates >= season_dates[0]) & (yr_dates <= season_dates[-1])
            yr_block = county_block[yr_mask]
            coverage = len(yr_block) / max(1, len(season_dates))
            if coverage < min_coverage_frac:
                n_dropped_coverage += 1
                sc.tick(extra=f"geoid={geoid} year={yr} (cov={coverage:.2f})")
                continue

            cdl_row = cdl_by_year_geoid.get((yr, geoid))
            if cdl_row is None and cdl_by_year_geoid:
                # Fall back to the nearest available year for this county.
                same_county = [
                    (abs(y - yr), y) for (y, g) in cdl_by_year_geoid.keys()
                    if g == geoid
                ]
                if same_county:
                    same_county.sort()
                    cdl_row = cdl_by_year_geoid[(same_county[0][1], geoid)]

            historical_mean = float(
                hist_mean_by_year.get(yr, pd.Series(dtype="float64"))
                .get(geoid, np.nan)
            )
            if not np.isfinite(historical_mean):
                # First training year for a county — use observed label as a
                # last-resort proxy. Safe because the model never sees this
                # static feature for the *target year* it's being asked to
                # predict at inference (we recompute per inference year).
                historical_mean = float(label)

            static_row = _build_static_row(
                geoid=geoid,
                year=yr,
                county_row=county,
                cdl_row=cdl_row,
                historical_mean=historical_mean,
            )

            try:
                tgt_ts, past_ts, future_ts = _build_series_for_county_year(
                    geoid=geoid,
                    year=yr,
                    weather_block=yr_block,
                    past_cols=past_cols,
                    season_dates=season_dates,
                    season_end_doy=season_end_doy,
                    label=float(label),
                    static_row=static_row,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "series-build failed for geoid=%s year=%d: %s",
                    geoid, yr, exc,
                )
                sc.tick(extra=f"geoid={geoid} year={yr} (build-fail)")
                continue

            target_series.append(tgt_ts)
            past_series.append(past_ts)
            future_series.append(future_ts)
            static_rows.append(static_row)
            index_rows.append({
                "geoid": geoid,
                "year": yr,
                "state_fips": geoid[:2],
                "label": float(label),
                "label_present": (yr, geoid) in nass_by_year_geoid,
                "coverage": float(coverage),
            })
            sc.tick()

    static_df = pd.DataFrame(static_rows, columns=list(STATIC_COVARIATE_COLS))
    index_df = pd.DataFrame(index_rows)

    bundle = TrainingBundle(
        target_series=target_series,
        past_covariates=past_series,
        future_covariates=future_series,
        static_covariates=static_df,
        series_index=index_df,
        past_covariate_cols=list(past_cols),
        static_covariate_cols=list(STATIC_COVARIATE_COLS),
    )

    logger.info(
        "bundle assembled: n_series=%d  dropped(label)=%d  dropped(cov)=%d  "
        "dropped(no-wx)=%d  past_cols=%d  future_cols=%d  static_cols=%d",
        bundle.n_series,
        n_dropped_label,
        n_dropped_coverage,
        n_dropped_no_weather,
        len(past_cols),
        len(FUTURE_COVARIATE_COLS),
        len(STATIC_COVARIATE_COLS),
    )

    if not index_df.empty:
        per_year = (
            index_df.groupby("year").size().to_dict()
        )
        logger.info("series per year: %s", per_year)
        logger.info(
            "[2025-leak-guard] max year in bundle = %d (must be ≤ %d)",
            int(index_df["year"].max()), MAX_TRAIN_YEAR,
        )
        # Belt-and-suspenders.
        assert int(index_df["year"].max()) <= MAX_TRAIN_YEAR, \
            "2025 leaked into training bundle"
    return bundle


def build_inference_dataset(
    states: Sequence[str] | None = None,
    target_year: int = 2025,
    season_start_doy: int = DEFAULT_SEASON_START_DOY,
    season_end_doy: int = DEFAULT_SEASON_END_DOY,
    include_sentinel: bool = False,
    include_smap: bool = True,
    refresh: bool = False,
    history_start_year: int = MIN_TRAIN_YEAR,
    history_end_year: int = MAX_TRAIN_YEAR,
    allow_download: bool = False,
    max_fetch_workers: int = 4,
) -> TrainingBundle:
    """Assemble a bundle for **inference** at a future year (default 2025).

    Differences from :func:`build_training_dataset`:

    - Pulls weather for ``target_year`` only (current season).
    - Uses per-county historical-mean yield (from
      ``[history_start_year, history_end_year]``) as a numeric placeholder
      target, so the model encoder isn't fed zeros.
    - Skips the label-required filter (`require_label=False`).
    - Sets ``label_present=False`` on every row in the index.

    Returns a :class:`TrainingBundle` with ``n_series == n_counties`` (one
    series per county for the target year).
    """
    if int(target_year) < MIN_TRAIN_YEAR:
        raise ValueError(
            f"target_year {target_year} predates available history "
            f"(MIN_TRAIN_YEAR={MIN_TRAIN_YEAR})"
        )
    # No upper bound — 2025+ is allowed for inference.

    from engine.cdl import fetch_counties_cdl
    from engine.counties import load_counties
    from engine.nass import fetch_counties_nass_yields
    from engine.weather import fetch_counties_weather

    banner(
        f"INFERENCE BUILD  target_year={target_year}  history={history_start_year}-{history_end_year}",
        logger=logger,
    )
    counties = load_counties(states=states)
    logger.info("counties loaded: n=%d", len(counties))

    # Historical yields (strictly < target_year for the prior, drops 2025):
    w = max(1, int(max_fetch_workers))
    nass_history = fetch_counties_nass_yields(
        counties,
        start_year=history_start_year,
        end_year=min(history_end_year, MAX_TRAIN_YEAR),
        refresh=refresh,
        max_workers=w,
    )
    if (nass_history["year"].astype(int) == 2025).any():
        nass_history = nass_history[nass_history["year"].astype(int) != 2025]
    historical_mean = _historical_mean_yields(nass_history, target_year=target_year)
    logger.info(
        "historical mean yield: counties=%d  mean=%.2f bu/acre  "
        "(computed over years %d-%d, strictly < %d)",
        len(historical_mean),
        float(historical_mean.mean()) if len(historical_mean) else float("nan"),
        history_start_year, min(history_end_year, MAX_TRAIN_YEAR),
        target_year,
    )

    # Weather for the target year only.
    weather = fetch_counties_weather(
        counties,
        start_year=int(target_year),
        end_year=int(target_year),
        include_smap=include_smap,
        include_sentinel=include_sentinel,
        refresh=refresh,
        max_workers=w,
    )
    weather_season = _filter_to_growing_season(
        weather, start_doy=season_start_doy, end_doy=season_end_doy
    )
    past_cols = _select_past_covariate_cols(weather_season)
    logger.info("past-covariate columns (%d): %s", len(past_cols), past_cols)

    # CDL for the target year. Falls back to last available year in code.
    cdl: pd.DataFrame
    try:
        cdl = fetch_counties_cdl(counties, year=int(target_year), resolution=30,
                                 refresh=refresh,
                                 allow_download=allow_download).assign(year=int(target_year))
    except Exception as exc:  # noqa: BLE001
        logger.warning("CDL %d unavailable (%s); falling back to %d",
                       target_year, exc, MAX_TRAIN_YEAR)
        cdl = fetch_counties_cdl(counties, year=MAX_TRAIN_YEAR, resolution=30,
                                 refresh=refresh,
                                 allow_download=allow_download).assign(year=int(target_year))

    cdl_by_geoid = {str(r["geoid"]): r for _, r in cdl.iterrows()}

    target_series: list = []
    past_series: list = []
    future_series: list = []
    static_rows: list[dict[str, float]] = []
    index_rows: list[dict] = []

    sc = StepCounter(logger, total=len(counties), unit="counties", every=25,
                     prefix="inference-series")

    season_dates = pd.date_range(
        start=pd.Timestamp(year=int(target_year), month=1, day=1)
        + pd.Timedelta(days=season_start_doy - 1),
        end=pd.Timestamp(year=int(target_year), month=1, day=1)
        + pd.Timedelta(days=season_end_doy - 1),
        freq="D",
    )

    n_dropped = 0
    for _, county in counties.iterrows():
        geoid = str(county["geoid"])
        try:
            block = weather_season.xs(geoid, level="geoid", drop_level=False)
        except KeyError:
            n_dropped += 1
            sc.tick(extra=f"geoid={geoid} (no weather)")
            continue

        proxy_label = float(historical_mean.get(geoid,
                                                historical_mean.mean()
                                                if len(historical_mean) else 0.0))
        static_row = _build_static_row(
            geoid=geoid,
            year=int(target_year),
            county_row=county,
            cdl_row=cdl_by_geoid.get(geoid),
            historical_mean=proxy_label,
        )

        try:
            tgt_ts, past_ts, future_ts = _build_series_for_county_year(
                geoid=geoid,
                year=int(target_year),
                weather_block=block,
                past_cols=past_cols,
                season_dates=season_dates,
                season_end_doy=season_end_doy,
                label=proxy_label,
                static_row=static_row,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("inference series-build failed for geoid=%s: %s",
                           geoid, exc)
            n_dropped += 1
            sc.tick(extra=f"geoid={geoid} (build-fail)")
            continue

        target_series.append(tgt_ts)
        past_series.append(past_ts)
        future_series.append(future_ts)
        static_rows.append(static_row)
        index_rows.append({
            "geoid": geoid,
            "year": int(target_year),
            "state_fips": geoid[:2],
            "label": proxy_label,
            "label_present": False,
            "coverage": float(len(block) / max(1, len(season_dates))),
        })
        sc.tick()

    bundle = TrainingBundle(
        target_series=target_series,
        past_covariates=past_series,
        future_covariates=future_series,
        static_covariates=pd.DataFrame(static_rows, columns=list(STATIC_COVARIATE_COLS)),
        series_index=pd.DataFrame(index_rows),
        past_covariate_cols=list(past_cols),
        static_covariate_cols=list(STATIC_COVARIATE_COLS),
    )
    logger.info(
        "inference bundle: n_series=%d  dropped=%d  past_cols=%d",
        bundle.n_series, n_dropped, len(past_cols),
    )
    return bundle


# ---------------------------------------------------------------------------
# CLI: inspect a bundle without writing parquet
# ---------------------------------------------------------------------------

def _summarize_bundle(bundle: TrainingBundle) -> str:
    if bundle.n_series == 0:
        return "bundle is empty"
    lines = [
        f"n_series:            {bundle.n_series}",
        f"past_cov columns:   {len(bundle.past_covariate_cols)} -> {bundle.past_covariate_cols[:6]}{'...' if len(bundle.past_covariate_cols) > 6 else ''}",
        f"future_cov columns: {len(FUTURE_COVARIATE_COLS)} -> {list(FUTURE_COVARIATE_COLS)}",
        f"static columns:     {len(bundle.static_covariate_cols)} -> {bundle.static_covariate_cols}",
    ]
    if not bundle.series_index.empty:
        per_year = bundle.series_index.groupby("year").size()
        per_state = bundle.series_index.groupby("state_fips").size()
        lines.append(f"series per year:    {per_year.to_dict()}")
        lines.append(f"series per state:   {per_state.to_dict()}")
        labels = bundle.series_index["label"]
        lines.append(
            f"label range:        min={labels.min():.1f}  "
            f"max={labels.max():.1f}  mean={labels.mean():.1f}  "
            f"sd={labels.std():.1f}"
        )
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the assembled Darts bundle without launching training."
    )
    parser.add_argument("--states", nargs="+", default=None, metavar="STATE",
                        help="One or more state names / 2-digit FIPS. "
                             "Omit for all 5 target states.")
    parser.add_argument("--start", type=int, default=MIN_TRAIN_YEAR,
                        help=f"Start year (default {MIN_TRAIN_YEAR}, min {MIN_TRAIN_YEAR}).")
    parser.add_argument("--end", type=int, default=MAX_TRAIN_YEAR,
                        help=f"End year (default {MAX_TRAIN_YEAR}, "
                             f"max {MAX_TRAIN_YEAR} — 2025 is the strict holdout).")
    parser.add_argument("--inference-year", type=int, default=None,
                        help="If set, build an inference bundle for this "
                             "year (instead of a training bundle). Use 2025 "
                             "for the deliverable forecast inputs.")
    parser.add_argument("--no-smap", action="store_true",
                        help="Skip SMAP soil moisture columns.")
    parser.add_argument("--include-sentinel", action="store_true",
                        help="Include Sentinel-2 NDVI/NDWI (slow, needs "
                             "[sentinel] extra installed).")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download of all underlying sources.")
    parser.add_argument("--allow-download", action="store_true",
                        help="Permit CDL raster downloads from USDA when a "
                             "year is missing from the data root. Without "
                             "this flag the engine stays in strict mode and "
                             "skips missing CDL years (you'll see warnings).")
    parser.add_argument("--out-stats", type=Path, default=None,
                        help="Optional: write the series_index dataframe to a "
                             "parquet/CSV for inspection.")
    parser.add_argument(
        "--save-last-bundle",
        action="store_true",
        help="After a training build, save the Darts TrainingBundle to "
             "<data_root>/derived/bundles/last_training_bundle.pkl (+ .meta.json). "
             "hack26-train reuses this by default to skip re-pulling weather/CDL.",
    )
    parser.add_argument(
        "--save-bundle",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save the training bundle to PATH.pkl with PATH.meta.json. "
             "Takes precedence over --save-last-bundle if both are set.",
    )
    parser.add_argument(
        "--max-fetch-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel I/O for weather, CDL years, and NASS states. "
             "Use 1 for a fully sequential fetch.",
    )
    add_cli_logging_args(parser)
    args = parser.parse_args(argv)

    log_path = apply_cli_logging_args(args, tag="dataset")
    log_environment(logger)
    logger.info("rotated log file: %s", log_path)

    try:
        if args.inference_year is not None:
            bundle = build_inference_dataset(
                states=args.states,
                target_year=args.inference_year,
                include_sentinel=args.include_sentinel,
                include_smap=not args.no_smap,
                refresh=args.refresh,
                allow_download=args.allow_download,
                max_fetch_workers=int(args.max_fetch_workers),
            )
        else:
            bundle = build_training_dataset(
                states=args.states,
                start_year=args.start,
                end_year=args.end,
                include_sentinel=args.include_sentinel,
                include_smap=not args.no_smap,
                refresh=args.refresh,
                allow_download=args.allow_download,
                max_fetch_workers=int(args.max_fetch_workers),
            )
    except ValueError as exc:
        logger.error("dataset build refused: %s", exc)
        return 2

    summary = _summarize_bundle(bundle)
    for line in summary.splitlines():
        logger.info(line)

    if args.out_stats is not None:
        args.out_stats.parent.mkdir(parents=True, exist_ok=True)
        suf = args.out_stats.suffix.lower()
        if suf == ".parquet":
            bundle.series_index.to_parquet(args.out_stats, index=False)
        elif suf == ".csv":
            bundle.series_index.to_csv(args.out_stats, index=False)
        else:
            logger.error("unsupported --out-stats suffix: %s "
                         "(use .parquet or .csv)", suf)
            return 2
        logger.info("wrote series-index summary to %s", args.out_stats)

    save_pkl: Path | None = args.save_bundle
    if save_pkl is None and args.save_last_bundle:
        save_pkl = default_last_training_bundle_path()
    if save_pkl is not None and args.inference_year is not None:
        logger.warning(
            "--save-bundle / --save-last-bundle only apply to training builds; "
            "inference build — skipping bundle save"
        )
        save_pkl = None

    if save_pkl is not None:
        from engine.counties import _resolve_states

        fips = _resolve_states(args.states)
        save_training_bundle(
            bundle,
            save_pkl,
            states_fips=fips,
            start_year=int(args.start),
            end_year=int(args.end),
            include_sentinel=args.include_sentinel,
            include_smap=not args.no_smap,
        )
        logger.info("wrote training bundle: %s", save_pkl)
        logger.info("wrote bundle metadata:  %s", _meta_path_for_pickle(save_pkl))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
