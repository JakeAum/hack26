"""Combine per-year hack26-forecast outputs into one Excel workbook.

Usage
-----
    python software/scripts/combine_forecasts.py \
        --inputs ~/hack26/data/derived/forecasts/forecast_2024.parquet \
                 ~/hack26/data/derived/forecasts/forecast_2025.parquet \
        --out    ~/hack26/data/derived/forecasts/corn_forecast_2024_2025.xlsx

For each input parquet, the script also picks up the sibling files
``<stem>_per_county_model.parquet`` and ``<stem>_per_county_analog.parquet``
(written by ``hack26-forecast``) and stacks them across years.

The resulting workbook has three sheets:

    state_forecasts      area-weighted state-level cone (one row per state x date x year)
    county_model_cone    per-county TFT predictive cone (model_p10 / p50 / p90)
    county_analog_cone   per-county analog-year baseline cone (yield_p25 / p50 / p75 / mean)

Requires ``openpyxl`` (already pulled in by pandas[excel]; install with
``pip install openpyxl`` if missing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _load_with_year(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "target_year" not in df.columns:
        # County-level outputs don't carry target_year; infer from filename.
        stem = path.stem
        for tok in stem.split("_"):
            if tok.isdigit() and len(tok) == 4:
                df = df.assign(target_year=int(tok))
                break
    return df


def _stack(paths: list[Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            print(f"  skip (missing): {p}", file=sys.stderr)
            continue
        df = _load_with_year(p)
        pieces.append(df)
        print(f"  loaded {p.name:60s} rows={len(df):>6d}")
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _order_state(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    front = [c for c in (
        "target_year", "forecast_date", "as_of",
        "state_fips", "state_name",
        "model_p10", "model_p50", "model_p90",
        "analog_p25", "analog_p50", "analog_p75", "analog_mean",
        "nass_baseline_yield", "nass_baseline_status", "nass_baseline_year",
    ) if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    out = df[front + rest].copy()
    out = out.sort_values(
        ["target_year", "state_fips", "forecast_date"]
    ).reset_index(drop=True)
    return out


def _order_county(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    front = [c for c in (
        "target_year", "forecast_date", "as_of",
        "geoid", "state_fips", "county_name",
    ) if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    out = df[front + rest].copy()
    sort_cols = [c for c in ("target_year", "geoid", "forecast_date") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="State-level forecast parquets, e.g. forecast_2024.parquet forecast_2025.parquet",
    )
    parser.add_argument(
        "--out", required=True,
        help="Path to the combined .xlsx workbook to write.",
    )
    args = parser.parse_args(argv)

    state_paths = [Path(p).expanduser() for p in args.inputs]
    county_model_paths = [
        p.with_name(p.stem + "_per_county_model" + p.suffix) for p in state_paths
    ]
    county_analog_paths = [
        p.with_name(p.stem + "_per_county_analog" + p.suffix) for p in state_paths
    ]

    print("Stacking state-level forecasts:")
    state_df = _order_state(_stack(state_paths))
    print("Stacking per-county model cones:")
    cmodel_df = _order_county(_stack(county_model_paths))
    print("Stacking per-county analog cones:")
    canalog_df = _order_county(_stack(county_analog_paths))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        state_df.to_excel(xw, sheet_name="state_forecasts", index=False)
        cmodel_df.to_excel(xw, sheet_name="county_model_cone", index=False)
        canalog_df.to_excel(xw, sheet_name="county_analog_cone", index=False)

    print(
        f"\nWrote {out_path}"
        f"\n  state_forecasts    rows={len(state_df)}"
        f"\n  county_model_cone  rows={len(cmodel_df)}"
        f"\n  county_analog_cone rows={len(canalog_df)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
