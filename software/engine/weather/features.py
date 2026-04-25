"""Feature engineering on top of the merged daily weather frame.

All inputs are expected to be indexed by ``(date, geoid)`` — the canonical
shape produced by :mod:`engine.weather.core`. Every transform groups by
``geoid`` so one county's tail can never bleed into the next county's head
(which is what would happen with a naive ``df.rolling(...)`` over the whole
flat frame).

Functions:
    :func:`compute_gdd`           — corn GDD + per-field/year cumulative GDD.
    :func:`add_rolling_features`  — 7-day and 30-day per-county rolling means.
    :func:`build_annual_summary`  — one row per (geoid, year) with sane aggs.
"""

from __future__ import annotations

import pandas as pd

# Corn GDD parameters. Standard NCERA-13 values.
GDD_BASE_TEMP_C = 10.0
GDD_MAX_TEMP_C = 30.0


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _level_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    return df.index.get_level_values("date")


def _level_geoids(df: pd.DataFrame) -> pd.Index:
    return df.index.get_level_values("geoid")


# ---------------------------------------------------------------------------
# GDD
# ---------------------------------------------------------------------------

def compute_gdd(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``GDD`` and ``GDD_cumulative`` columns.

    GDD = ((min(T_max, 30) + max(T_min, 10)) / 2) − 10, clipped to ≥ 0.

    ``GDD_cumulative`` resets per ``(geoid, calendar year)`` so the running
    total reflects "growing season to date" within a single county-year — a
    naive cumsum would carry one county's December GDD into the next county's
    January, which is meaningless.

    No-op if either ``T2M_MAX`` or ``T2M_MIN`` is absent.
    """
    if "T2M_MAX" not in df.columns or "T2M_MIN" not in df.columns:
        return df

    out = df.copy()
    t_max = out["T2M_MAX"].clip(upper=GDD_MAX_TEMP_C)
    t_min = out["T2M_MIN"].clip(lower=GDD_BASE_TEMP_C)
    gdd = ((t_max + t_min) / 2.0) - GDD_BASE_TEMP_C
    out["GDD"] = gdd.clip(lower=0)

    geoids = _level_geoids(out)
    years = _level_dates(out).year
    out["GDD_cumulative"] = out.groupby([geoids, years])["GDD"].cumsum()
    return out


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (7, 30),
) -> pd.DataFrame:
    """Per-county rolling means for every numeric base column.

    Suffix is ``_{N}d_avg`` (e.g. ``T2M_7d_avg``). Already-suffixed columns
    are skipped on re-runs so this is idempotent. The groupby on ``geoid``
    is essential — without it, the last 7 days of county A's series would
    contaminate the first 7 of county B.
    """
    if df.empty:
        return df.copy()

    base_cols = [
        c for c in df.columns
        if not any(c.endswith(f"_{w}d_avg") for w in windows)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    out = df.copy()

    for col in base_cols:
        for w in windows:
            new_col = f"{col}_{w}d_avg"
            out[new_col] = (
                out.groupby(level="geoid")[col]
                .transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            )
    return out


# ---------------------------------------------------------------------------
# Annual summary
# ---------------------------------------------------------------------------

def _frost_days_annual(s: pd.Series) -> float:
    """FROST_DAYS in NASA POWER is a *monthly* count repeated daily, so a
    plain sum over a year over-counts by ~30x. Average within each month and
    then sum the monthly means to get a real annual frost-day total."""
    if s.empty:
        return float("nan")
    # Drop the geoid level so resample sees a clean DatetimeIndex.
    s = s.reset_index(level="geoid", drop=True)
    return float(s.resample("ME").mean().sum())


def build_annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(geoid, year)`` summarizing the daily frame.

    Only columns actually present in ``df`` get aggregated, so this is safe
    to call on partial frames (e.g. POWER-only, no Sentinel).
    """
    if df.empty:
        return pd.DataFrame()

    base = df[[c for c in df.columns
               if not c.endswith(("_7d_avg", "_30d_avg"))]]

    aggs: dict[str, tuple[str, str | callable]] = {}

    if "PRECTOTCORR" in base.columns:
        aggs["PRECTOTCORR_total_mm"] = ("PRECTOTCORR", "sum")
        aggs["PRECTOTCORR_avg_mm_day"] = ("PRECTOTCORR", "mean")

    if "RH2M" in base.columns:
        aggs["RH2M_avg_pct"] = ("RH2M", "mean")
        aggs["RH2M_min_pct"] = ("RH2M", "min")
        aggs["RH2M_max_pct"] = ("RH2M", "max")

    if "T2MDEW" in base.columns:
        aggs["T2MDEW_avg_C"] = ("T2MDEW", "mean")

    if "EVPTRNS" in base.columns:
        aggs["EVPTRNS_total_mm"] = ("EVPTRNS", "sum")
        aggs["EVPTRNS_avg_mm_day"] = ("EVPTRNS", "mean")

    for col in ("GWETROOT", "GWETTOP", "GWETPROF"):
        if col in base.columns:
            aggs[f"{col}_avg"] = (col, "mean")
            aggs[f"{col}_min"] = (col, "min")
            aggs[f"{col}_max"] = (col, "max")

    if "SMAP_surface_sm_m3m3" in base.columns:
        aggs["SMAP_surface_sm_avg"] = ("SMAP_surface_sm_m3m3", "mean")
        aggs["SMAP_surface_sm_min"] = ("SMAP_surface_sm_m3m3", "min")
        aggs["SMAP_surface_sm_max"] = ("SMAP_surface_sm_m3m3", "max")

    if "T2M" in base.columns:
        aggs["T2M_avg_C"] = ("T2M", "mean")
        aggs["T2M_min_C"] = ("T2M", "min")
        aggs["T2M_max_C"] = ("T2M", "max")
    if "T2M_MAX" in base.columns:
        aggs["T2M_MAX_avg_C"] = ("T2M_MAX", "mean")
    if "T2M_MIN" in base.columns:
        aggs["T2M_MIN_avg_C"] = ("T2M_MIN", "mean")
    if "TS" in base.columns:
        aggs["TS_avg_C"] = ("TS", "mean")
    if "T10M" in base.columns:
        aggs["T10M_avg_C"] = ("T10M", "mean")

    if "GDD" in base.columns:
        aggs["GDD_total"] = ("GDD", "sum")
        aggs["GDD_avg"] = ("GDD", "mean")

    if "NDVI" in base.columns:
        aggs["NDVI_avg"] = ("NDVI", "mean")
        aggs["NDVI_min"] = ("NDVI", "min")
        aggs["NDVI_max"] = ("NDVI", "max")
    if "NDWI" in base.columns:
        aggs["NDWI_avg"] = ("NDWI", "mean")

    # Build the per-(geoid, year) aggregate. We don't lean on
    # `groupby(level="geoid").resample("YE")` because pandas' API for that has
    # been a moving target across versions; a calendar-year column off the
    # date level is rock-solid and produces the same result.
    years = _level_dates(base).year
    geoids = _level_geoids(base)
    keyed = base.assign(_year=years.values)
    grouped = keyed.groupby([geoids, "_year"])

    named = {alias: pd.NamedAgg(column=col, aggfunc=fn)
             for alias, (col, fn) in aggs.items()}
    out = grouped.agg(**named) if named else grouped.size().to_frame("rows")

    if "FROST_DAYS" in base.columns:
        out["FROST_DAYS_annual"] = (
            base["FROST_DAYS"]
            .groupby([geoids, years])
            .apply(_frost_days_annual)
        )

    out.index.names = ["geoid", "year"]
    return out
