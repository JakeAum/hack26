"""USDA NASS Quick Stats — corn-for-grain yield (county) and state in-season
forecasts / finals.

Satisfies the SPEC §2 contract for the NASS source (join key ``geoid``)::

    fetch_county_nass_yields(geoid, geometry, start_year, end_year) -> DataFrame
    fetch_counties_nass_yields(counties, start_year, end_year) -> DataFrame

``geometry`` is accepted for a uniform call signature and ignored — NASS is
not spatially queried, only FIPS- and year-keyed.

**Setup.** Register a free key at <https://quickstats.nass.usda.gov/>, then::

    $env:NASS_API_KEY = "<key>"   # PowerShell
    export NASS_API_KEY=<key>      # bash

**CLI.**
    python -m engine.nass --counties --states Iowa --start 2020 --end 2024
    python -m engine.nass --state-forecasts --states 19 31 --start 2018 --end 2024
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import geopandas as gpd
import pandas as pd
import requests

from ._cache import county_yields_path, state_forecasts_path

# --- Quick Stats: corn for grain, yield in bu/acre
#
# NOTE (2026 NASS schema migration): NASS Quick Stats has consolidated all
# corn ``class_desc`` values down to ``ALL CLASSES`` / ``TRADITIONAL OR
# INDIAN`` and now expresses the grain-vs-silage distinction *only* via
# ``util_practice_desc``. Sending the legacy ``class_desc=GRAIN`` returns
# ``HTTP 400 {"error":["bad request - invalid query"]}``. The slice we want
# (``CORN, GRAIN - YIELD, MEASURED IN BU / ACRE`` in ``short_desc``) is now
# expressed as::
#
#     class_desc=ALL CLASSES, util_practice_desc=GRAIN
#
# The cached parquets from the previous schema are still scientifically
# correct (same scientific slice, same ``short_desc``); the cache key is
# ``(state_ansi, start_year, end_year)`` and not parameter-content-aware,
# so existing files do not need invalidation.
_COMMOD = "CORN"
_CLASS = "ALL CLASSES"
_STAT = "YIELD"
_UNIT = "BU / ACRE"
_PRODN = "ALL PRODUCTION PRACTICES"
_UTIL = "GRAIN"
_SOURCE = "SURVEY"

# County annual final only (NASS does not publish in-season at county).
_COUNTY_REF = "YEAR"
_FREQ = "ANNUAL"

# State: official monthly forecasts + the annual final in Quick Stats
_STATE_REFS: tuple[str, ...] = (
    "YEAR - AUG FORECAST",
    "YEAR - SEP FORECAST",
    "YEAR - OCT FORECAST",
    "YEAR - NOV FORECAST",
    "YEAR",
)

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"


def _validate_year_range(start_year: int, end_year: int) -> None:
    if start_year > end_year:
        raise ValueError(f"invalid year range: start_year={start_year} > end_year={end_year}")


def nass_api_key() -> str:
    k = (os.environ.get("NASS_API_KEY") or "").strip()
    if not k:
        raise OSError(
            "NASS_API_KEY is not set. Request a key at https://quickstats.nass.usda.gov/ "
            "and set the environment variable before calling engine.nass."
        )
    return k


def _parse_nass_value(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if s in ("(D)", "(X)", "(NA)", "(L)", "NA", "", "-"):
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _nass_redact_url(url: str) -> str:
    """Strip the ``key=...`` query param so URLs are safe to log."""
    import re
    return re.sub(r"(key=)[^&]+", r"\1***", url)


def nass_get(params: dict[str, str | int]) -> list[dict[str, Any]]:
    """Call Quick Stats ``api_GET``; return the ``data`` list (or []).

    Returns an **empty list** when NASS responds 4xx with the well-known
    "the query exceeded the limit of 50000 records or no data" message — that
    is operational, not an error, and downstream callers already cope with
    empty frames.

    For any other 4xx/5xx, raises :class:`requests.HTTPError` after first
    logging the response body and a redacted URL, so failures on the AWS box
    are diagnosable from the rotating log file.
    """
    q: dict[str, Any] = {
        "key": nass_api_key(),
        "format": "JSON",
        **params,
    }
    r = requests.get(NASS_API_URL, params=q, timeout=120)

    if not r.ok:
        body = ""
        try:
            body = r.text[:500]
        except Exception:  # noqa: BLE001
            body = ""
        # NASS uses HTTP 400 with a JSON ``{"error":["no data..."]}`` body
        # for both "no rows match" and "result set too large". Treat the
        # benign no-data case as an empty result so cold pulls don't crash
        # over years/states that legitimately have nothing.
        lowered = body.lower()
        if r.status_code == 400 and (
            '"no data"' in lowered
            or "no data " in lowered
            or "no records" in lowered
        ):
            print(
                f"[nass] empty response (HTTP 400 'no data') for "
                f"{_nass_redact_url(r.url)}",
                file=sys.stderr,
            )
            return []
        # Surface a useful diagnostic before raising — NASS error bodies are
        # usually short JSON like ``{"error": ["..."]}``.
        print(
            f"[nass] HTTP {r.status_code} for {_nass_redact_url(r.url)}\n"
            f"[nass] body: {body!r}",
            file=sys.stderr,
        )
        r.raise_for_status()

    j: Any = r.json()
    if not isinstance(j, dict) or "data" not in j:
        return []
    data = j.get("data")
    return data if isinstance(data, list) else []


def _geoid_from_row(row: dict[str, Any]) -> str:
    st = str(row.get("state_ansi", "")).zfill(2)
    co = str(row.get("county_ansi", "")).zfill(3)
    if len(st) != 2 or len(co) != 3 or not st.isdigit() or not co.isdigit():
        return ""
    return st + co


def _is_other_counties_row(row: dict[str, Any]) -> bool:
    name = (row.get("county_name") or "").upper()
    code = str(row.get("county_code", "")).zfill(3)
    if code == "998":
        return True
    if "OTHER" in name and "COUNT" in name:
        return True
    return False


def _normalize_county_yields(
    raw: list[dict[str, Any]], allowed_geoids: set[str] | None
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in raw:
        if _is_other_counties_row(rec):
            continue
        gid = _geoid_from_row(rec)
        if not gid or (allowed_geoids is not None and gid not in allowed_geoids):
            continue
        y = _parse_nass_value(rec.get("Value"))
        if y is None:
            continue
        year = int(rec.get("year", 0) or 0)
        if year <= 0:
            continue
        rows.append(
            {
                "geoid": gid,
                "state_ansi": str(rec.get("state_ansi", "")).zfill(2),
                "county_ansi": str(rec.get("county_ansi", "")).zfill(3),
                "county_name": rec.get("county_name"),
                "state_alpha": rec.get("state_alpha"),
                "state_name": rec.get("state_name"),
                "year": year,
                "reference_period_desc": rec.get("reference_period_desc"),
                "short_desc": rec.get("short_desc"),
                "statisticcat_desc": rec.get("statisticcat_desc"),
                "unit_desc": _UNIT,
                "agg_level_desc": "COUNTY",
                "nass_value": y,
                "load_time": rec.get("load_time"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "geoid",
                "state_ansi",
                "county_ansi",
                "county_name",
                "state_alpha",
                "state_name",
                "year",
                "reference_period_desc",
                "short_desc",
                "statisticcat_desc",
                "unit_desc",
                "agg_level_desc",
                "nass_value",
                "load_time",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["geoid", "year"]).drop_duplicates(
        subset=["geoid", "year", "reference_period_desc"], keep="last"
    )
    return df.reset_index(drop=True)


def _county_yields_base_params(
    state_ansi: str, start_year: int, end_year: int
) -> dict[str, str | int]:
    return {
        "source_desc": _SOURCE,
        "commodity_desc": _COMMOD,
        "class_desc": _CLASS,
        "prodn_practice_desc": _PRODN,
        "util_practice_desc": _UTIL,
        "statisticcat_desc": _STAT,
        "unit_desc": _UNIT,
        "agg_level_desc": "COUNTY",
        "state_ansi": str(state_ansi).zfill(2),
        "domain_desc": "TOTAL",
        "reference_period_desc": _COUNTY_REF,
        "freq_desc": _FREQ,
        "year__GE": start_year,
        "year__LE": end_year,
    }


def _pull_county_state(
    state_ansi: str,
    start_year: int,
    end_year: int,
    allowed_geoids: set[str] | None,
    refresh: bool,
) -> pd.DataFrame:
    path = county_yields_path(state_ansi, start_year, end_year)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        if allowed_geoids is not None and not df.empty:
            df = df[df["geoid"].isin(allowed_geoids)].copy()
        return df.reset_index(drop=True)

    raw = nass_get(_county_yields_base_params(state_ansi, start_year, end_year))
    df = _normalize_county_yields(raw, None)
    if not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    if allowed_geoids is not None and not df.empty:
        df = df[df["geoid"].isin(allowed_geoids)].copy()
    return df.reset_index(drop=True)


def fetch_county_nass_yields(
    geoid: str,
    geometry: object = None,
    start_year: int = 2000,
    end_year: int = 2024,
    refresh: bool = False,
) -> pd.DataFrame:
    """Corn-for-grain **final** county yield (bu/acre) from NASS, one row per
    year. ``reference_period`` is the annual final (``YEAR``).

    ``geometry`` is ignored (NASS is not spatially queried).
    """
    _validate_year_range(start_year, end_year)
    if not geoid or len(geoid) != 5 or not geoid.isdigit():
        raise ValueError("geoid must be a 5-digit county FIPS string, e.g. 19169")
    state_ansi = geoid[:2]
    return _pull_county_state(
        state_ansi, start_year, end_year, {geoid}, refresh=refresh
    )


def fetch_counties_nass_yields(
    counties: gpd.GeoDataFrame,
    start_year: int = 2000,
    end_year: int = 2024,
    refresh: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Corn-for-grain **final** county yields for all rows in ``counties``
    (must have ``geoid``). One API + cache per distinct state; filters to
    requested geoids. Excludes NASS ``OTHER (COMBINED) COUNTIES`` aggregate rows.

    When ``max_workers`` > 1, state-level pulls (each its own API + cache) run
    concurrently (up to 5 for the five project states).
    """
    _validate_year_range(start_year, end_year)
    if "geoid" not in counties.columns:
        raise ValueError("counties GeoDataFrame must have a 'geoid' column")
    if counties.empty:
        return _normalize_county_yields([], None)

    geoids = {str(g).zfill(5) for g in counties["geoid"]}
    by_state: dict[str, set[str]] = {}
    for g in geoids:
        by_state.setdefault(g[:2], set()).add(g)

    state_items = sorted(by_state.items())
    if not state_items:
        return _normalize_county_yields([], None)

    if max_workers <= 1 or len(state_items) == 1:
        pieces = [
            _pull_county_state(st, start_year, end_year, gset, refresh=refresh)
            for st, gset in state_items
        ]
    else:
        def _one(item: tuple[str, set[str]]) -> pd.DataFrame:
            st, gset = item
            return _pull_county_state(
                st, start_year, end_year, gset, refresh=refresh,
            )

        w = min(max(1, int(max_workers)), len(state_items))
        with ThreadPoolExecutor(max_workers=w) as ex:
            pieces = list(ex.map(_one, state_items))

    if not pieces:
        return _normalize_county_yields([], None)
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["geoid", "year"]).reset_index(drop=True)


# --- state forecasts (national replacement baseline at state level) ---


def _normalize_state_forecasts(
    raw: list[dict[str, Any]], state_ansi: str
) -> pd.DataFrame:
    want = set(_STATE_REFS)
    st = str(state_ansi).zfill(2)
    rows: list[dict[str, Any]] = []
    for rec in raw:
        if str(rec.get("state_ansi", "")).zfill(2) != st:
            continue
        if str(rec.get("agg_level_desc", "")).upper() != "STATE":
            continue
        ref = rec.get("reference_period_desc")
        if ref not in want:
            continue
        y = _parse_nass_value(rec.get("Value"))
        if y is None:
            continue
        year = int(rec.get("year", 0) or 0)
        if year <= 0:
            continue
        rows.append(
            {
                "geoid": st + "000",  # synthetic 5-char state key, e.g. 19000 (not a county FIPS)
                "state_ansi": st,
                "state_alpha": rec.get("state_alpha"),
                "state_name": rec.get("state_name"),
                "year": year,
                "reference_period_desc": ref,
                "short_desc": rec.get("short_desc"),
                "statisticcat_desc": rec.get("statisticcat_desc"),
                "unit_desc": _UNIT,
                "agg_level_desc": "STATE",
                "nass_value": y,
                "load_time": rec.get("load_time"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "geoid",
                "state_ansi",
                "state_alpha",
                "state_name",
                "year",
                "reference_period_desc",
                "short_desc",
                "statisticcat_desc",
                "unit_desc",
                "agg_level_desc",
                "nass_value",
                "load_time",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["state_ansi", "year", "reference_period_desc"]
    ).drop_duplicates(
        subset=["state_ansi", "year", "reference_period_desc"], keep="last"
    )
    return df.reset_index(drop=True)


def _state_forecast_params(
    state_ansi: str, start_year: int, end_year: int
) -> dict[str, str | int]:
    # Pull all YIELD rows for the state + year range; filter reference_period in Python
    return {
        "source_desc": _SOURCE,
        "commodity_desc": _COMMOD,
        "class_desc": _CLASS,
        "prodn_practice_desc": _PRODN,
        "util_practice_desc": _UTIL,
        "statisticcat_desc": _STAT,
        "unit_desc": _UNIT,
        "agg_level_desc": "STATE",
        "state_ansi": str(state_ansi).zfill(2),
        "domain_desc": "TOTAL",
        "freq_desc": "ANNUAL",
        "year__GE": start_year,
        "year__LE": end_year,
    }


def _pull_state_forecasts(
    state_ansi: str, start_year: int, end_year: int, refresh: bool
) -> pd.DataFrame:
    path = state_forecasts_path(state_ansi, start_year, end_year)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = nass_get(_state_forecast_params(state_ansi, start_year, end_year))
    df = _normalize_state_forecasts(raw, state_ansi)
    if not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


def fetch_nass_state_corn_forecasts(
    state_fips_list: Sequence[str] | None = None,
    start_year: int = 2015,
    end_year: int = 2024,
    refresh: bool = False,
) -> pd.DataFrame:
    """State-level corn-for-grain **yield** (forecasts and annual final) from
    Quick Stats, aligned with the monthly *Crop Production* release labels.

    ``geoid`` is a synthetic ``<state>000`` (e.g. ``19000``) — state-only, not
    a county FIPS.

    If ``state_fips_list`` is None, all five target states (from
    ``engine.counties``) are queried.
    """
    _validate_year_range(start_year, end_year)
    from engine.counties import TARGET_STATES

    states = list(TARGET_STATES.keys()) if state_fips_list is None else [
        str(s).zfill(2) for s in state_fips_list
    ]
    parts: list[pd.DataFrame] = []
    for st in states:
        parts.append(_pull_state_forecasts(st, start_year, end_year, refresh))
    if not parts:
        return _normalize_state_forecasts([], "19")
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["state_ansi", "year", "reference_period_desc"]).reset_index(
        drop=True
    )


# --- CLI ---


def _main(argv: list[str] | None = None) -> int:
    from pathlib import Path

    from engine.counties import load_counties

    parser = argparse.ArgumentParser(
        description="USDA NASS Quick Stats: corn-for-grain yields (county) and state forecasts."
    )
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument(
        "--counties",
        action="store_true",
        help="Fetch final county-level yields (bu/ac) for the selected states.",
    )
    sub.add_argument(
        "--state-forecasts",
        action="store_true",
        help="Fetch state-level in-season + final yield rows (Aug/Sep/Oct/Nov + YEAR).",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        metavar="STATE",
        help="State names or 2-digit FIPS. For --counties, subset catalog; for "
        "--state-forecasts, which state series to query.",
    )
    parser.add_argument(
        "--start", type=int, default=2018, help="Start year (inclusive)."
    )
    parser.add_argument(
        "--end", type=int, default=2024, help="End year (inclusive)."
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore parquet cache, re-pull from NASS."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write result (.parquet or .csv).",
    )
    args = parser.parse_args(argv)

    if args.out is not None and args.out.suffix.lower() not in (".parquet", ".csv"):
        parser.error("--out must end with .parquet or .csv")
    if args.start > args.end:
        parser.error(f"--start ({args.start}) must be <= --end ({args.end})")

    try:
        nass_api_key()  # fail fast
    except OSError as exc:
        parser.error(str(exc))

    if args.counties:
        gdf = load_counties(states=args.states, refresh=False)
        df = fetch_counties_nass_yields(
            gdf, start_year=args.start, end_year=args.end, refresh=args.refresh
        )
        n = len(gdf)
        m = len(df)
        st_list = (
            sorted(gdf["state_fips"].unique().astype(str).tolist()) if n else []
        )
        print(
            f"county yield rows: {m}  (from {n} catalog counties, "
            f"state_fips {st_list!r}, {args.start}-{args.end})",
            file=sys.stderr,
        )
    else:
        st_fips: Sequence[str] | None
        if args.states is not None:
            s = load_counties(states=args.states, refresh=False)
            st_fips = sorted(s["state_fips"].unique().astype(str).str.zfill(2).tolist())
        else:
            st_fips = None
        df = fetch_nass_state_corn_forecasts(
            state_fips_list=st_fips,
            start_year=args.start,
            end_year=args.end,
            refresh=args.refresh,
        )
        print(
            f"state forecast rows: {len(df)}  "
            f"({args.start}-{args.end}, "
            f"states {list(df['state_ansi'].unique()) if not df.empty else '—'})",
            file=sys.stderr,
        )
    with pd.option_context("display.max_rows", 12, "display.width", 120):
        print(df)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix.lower() == ".parquet":
            df.to_parquet(args.out, index=False)
        else:
            df.to_csv(args.out, index=False)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())