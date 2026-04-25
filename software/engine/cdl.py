"""CDL (Cropland Data Layer) source — corn mask + class breakdown per county.

Wraps USDA NASS Cropland Data Layer national rasters and turns them into the
SPEC §2 contract: ``fetch(geoid, geometry, ...) -> pd.DataFrame``, keyed on
``geoid`` so the result joins cleanly against the County Catalog.

Data discovery (in order):
    1. ``$HACK26_CDL_DATA_DIR/{year}_{res}m_cdls.tif``   (env override)
    2. ``~/hack26/data/{year}_{res}m_cdls.tif``          (EFS mount per README)
    3. ``~/.hack26/cdl/{year}_{res}m_cdls.tif``          (our download cache)
    4. download zip → extract → 3.

The first two are how the AWS workshop machine ships the pre-extracted 14.9 GB
2025 10 m raster from EFS; we never want to re-pull 10 GB if it's already on
disk. Download only kicks in when nothing is found, which is the only path
that works for prior-year (≤2024 30 m) coverage.

Public surface:
    load_cdl(year=2025, resolution=30, refresh=False) -> Path
    fetch_county_cdl(geoid, geometry, year=2025, resolution=30, src=None) -> pd.DataFrame
    fetch_counties_cdl(gdf, year=2025, resolution=30, refresh=False) -> pd.DataFrame

CLI:
    python -m engine.cdl                              # year=2025, all 5 states, 30 m
    python -m engine.cdl --year 2024 --resolution 30
    python -m engine.cdl --states Iowa Colorado
    python -m engine.cdl --download-only              # fetch + extract, skip masking
    python -m engine.cdl --out cdl_features.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

# CDL availability matrix — bumped as NASS publishes new years.
EARLIEST_YEAR = 2008
LATEST_YEAR = 2025
# 10 m generation only exists for the most recent two years; everything else is 30 m.
TENM_YEARS = {2024, 2025}

NASS_URL_TPL = (
    "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/"
    "{year}_{res}m_cdls.zip"
)
# Workshop mirror in us-east-1 — fast on the AWS sagemaker box, anonymous read.
WORKSHOP_S3_TPL = "s3://rayette.guru/workshop/{year}_{res}m_cdls.zip"

# CDL class codes we care about for corn-yield forecasting. Class 1 = corn-for-grain;
# 12 (sweet corn) and 13 (pop/orn corn) are different markets and not the
# replacement target, but we report them so callers can sanity-check.
# Full table: https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/meta.php
CDL_CORN_FOR_GRAIN = 1
CDL_SWEET_CORN = 12
CDL_POP_ORN_CORN = 13
CDL_SOYBEANS = 5

# Classes we treat as "non-cropland" when computing the cropland-only denominator.
# 0 is the raster nodata/background; 81-92 + 111-195 are water, developed, forest,
# wetlands, etc. Keeping this narrow because the per-county pixel histogram is
# returned in full anyway — callers can re-aggregate however they want.
NONCROPLAND_CLASSES = frozenset(
    [0]
    + list(range(63, 65))      # 63 Forest, 64 Shrubland
    + list(range(81, 93))      # 81 Clouds/No Data … 92 Aquaculture
    + list(range(111, 196))    # 111 Open Water … 195 Herbaceous Wetlands
)


# ---------------------------------------------------------------------------
# Cache + data discovery
# ---------------------------------------------------------------------------

def _data_root() -> Path:
    """The single source of truth for CDL data.

    Resolves to ``$HACK26_CDL_DATA_DIR`` if set, else ``~/hack26/data``. The
    directory MUST exist — engine code refuses to fall back to ``~/.hack26``
    or trigger a multi-GB download on the AWS workshop box. If it doesn't
    exist, raise ``FileNotFoundError`` so the operator knows the EFS mount
    is missing instead of quietly burning bandwidth.
    """
    env = os.environ.get("HACK26_CDL_DATA_DIR")
    root = Path(env) if env else Path.home() / "hack26" / "data"
    if not root.is_dir():
        raise FileNotFoundError(
            f"CDL data root not found: {root}\n"
            f"Engine refuses to fall back to ~/.hack26 or download "
            f"multi-GB rasters at runtime.\n"
            f"Fix: mount the EFS data dir at ~/hack26/data, or set "
            f"HACK26_CDL_DATA_DIR=/path/to/data."
        )
    return root


def _derived_dir() -> Path:
    """Writable subdir of the data root for our per-county parquet outputs.

    Lives under the same root as the rasters (``~/hack26/data/derived/``) so
    we never touch ``~/.hack26``. Auto-created on first use.
    """
    d = _data_root() / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _zip_path(year: int, resolution: int) -> Path:
    """Where ``--download-only`` writes a freshly-pulled zip — under the EFS
    data root, not ``~/.hack26``."""
    return _data_root() / f"{year}_{resolution}m_cdls.zip"


def _tif_basename(year: int, resolution: int) -> str:
    return f"{year}_{resolution}m_cdls.tif"


def _validate(year: int, resolution: int) -> None:
    if not EARLIEST_YEAR <= year <= LATEST_YEAR:
        raise ValueError(
            f"year {year} out of range; CDL national coverage is "
            f"{EARLIEST_YEAR}-{LATEST_YEAR}"
        )
    if resolution not in (10, 30):
        raise ValueError(f"resolution must be 10 or 30 (got {resolution})")
    if resolution == 10 and year not in TENM_YEARS:
        raise ValueError(
            f"10 m CDL only published for {sorted(TENM_YEARS)}; "
            f"use resolution=30 for {year}"
        )


def _find_existing_tif(year: int, resolution: int) -> Path | None:
    """Return the .tif under the data root if it exists, else ``None``.

    Single-location lookup — no fallback chain. The data root itself is
    validated by ``_data_root()``; this only reports whether the requested
    (year, resolution) raster has been pre-staged inside it.
    """
    candidate = _data_root() / _tif_basename(year, resolution)
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def _aws_cli_available() -> bool:
    return shutil.which("aws") is not None


def _download_via_aws_cli(s3_uri: str, target: Path) -> None:
    """Anonymous S3 fetch via the AWS CLI. Used on the workshop sagemaker box
    where the CLI is installed and an in-region S3 GET is faster than HTTPS."""
    print(f"[cdl] aws s3 cp --no-sign-request {s3_uri}", file=sys.stderr)
    tmp = target.with_suffix(".zip.partial")
    proc = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", s3_uri, str(tmp)],
        check=False,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"aws s3 cp failed (exit {proc.returncode}) for {s3_uri}")
    tmp.replace(target)


def _download_via_https(url: str, target: Path) -> None:
    """Streaming HTTPS download with a .partial sentinel so a Ctrl-C mid-pull
    doesn't leave a corrupt zip in the cache that we'd happily reuse later."""
    print(f"[cdl] downloading {url}", file=sys.stderr)
    tmp = target.with_suffix(".zip.partial")
    # Long timeout: NASS files are 1.6-9.8 GB.
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 << 20):  # 8 MiB
                if chunk:
                    fh.write(chunk)
    tmp.replace(target)


def _download_zip(year: int, resolution: int, source: str = "auto") -> Path:
    """Pull the national CDL zip into our cache. Idempotent: skips if present.

    ``source``:
      - ``"auto"``   : workshop S3 if (year=2025, res=10, aws CLI present), else NASS.
      - ``"workshop"``: force the s3://rayette.guru mirror (only valid for 2025 10 m).
      - ``"nass"``   : force NASS HTTPS.
    """
    target = _zip_path(year, resolution)
    if target.exists():
        return target

    use_workshop = source == "workshop" or (
        source == "auto" and year == 2025 and resolution == 10 and _aws_cli_available()
    )

    if use_workshop:
        if not (year == 2025 and resolution == 10):
            raise ValueError(
                f"workshop mirror only hosts 2025 10 m; got year={year} res={resolution}m"
            )
        _download_via_aws_cli(WORKSHOP_S3_TPL.format(year=year, res=resolution), target)
    else:
        _download_via_https(NASS_URL_TPL.format(year=year, res=resolution), target)

    # Guard against silently caching an HTML error page or truncated download.
    with zipfile.ZipFile(target) as zf:
        if not any(n.endswith(".tif") for n in zf.namelist()):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded file has no .tif inside: {target}")
    return target


def _extract_tif(zip_path: Path, year: int, resolution: int) -> Path:
    """Extract the .tif (and any sidecars: .tif.ovr, .tif.aux.xml, .tfw) into
    the data root. Returns the path to the .tif. Idempotent.

    We extract sidecars too because rasterio uses .ovr for fast overview reads
    and we don't want it silently regenerating multi-GB pyramids on first open.
    """
    out_dir = _data_root()
    tif_path = out_dir / _tif_basename(year, resolution)
    if tif_path.exists():
        return tif_path

    print(f"[cdl] extracting {zip_path.name} → {out_dir}", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n for n in zf.namelist()
            # CDL zips occasionally nest under a folder; strip the leading dir
            # and pull anything that looks raster-ish or like its sidecar.
            if n.endswith((".tif", ".tif.ovr", ".tif.aux.xml", ".tfw", ".aux"))
        ]
        if not any(m.endswith(".tif") for m in members):
            raise RuntimeError(f"no .tif inside {zip_path}")
        for m in members:
            # Flatten any nested directory layout.
            dest = out_dir / Path(m).name
            with zf.open(m) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 << 20)

    if not tif_path.exists():
        # Some years use a slightly different inner filename — fall back to the
        # first .tif we extracted and rename it to our canonical name.
        loose = next(out_dir.glob("*_cdls.tif"), None)
        if loose is None:
            raise RuntimeError(f"extraction succeeded but no *_cdls.tif found in {out_dir}")
        loose.rename(tif_path)
    return tif_path


# ---------------------------------------------------------------------------
# Public: locate the raster
# ---------------------------------------------------------------------------

def load_cdl(
    year: int = 2025,
    resolution: int = 30,
    refresh: bool = False,
    source: str = "auto",
    allow_download: bool = False,
) -> Path:
    """Return a path to the national CDL GeoTIFF for ``year`` at ``resolution`` m.

    Strict-mode discovery: the file MUST already live under ``_data_root()``
    (``$HACK26_CDL_DATA_DIR`` or ``~/hack26/data``). No ``~/.hack26`` fallback,
    no implicit downloads. If the raster is missing we raise
    ``FileNotFoundError`` so callers don't accidentally trigger a 1.6-9.8 GB
    pull from a hot path.

    Pass ``allow_download=True`` (or run the CLI with ``--allow-download``)
    to opt-in to fetching+extracting from NASS / the workshop S3 mirror;
    that path also writes into the data root, never ``~/.hack26``.
    """
    _validate(year, resolution)

    if refresh:
        # Only ever clobber files inside the data root.
        for p in (_zip_path(year, resolution), _data_root() / _tif_basename(year, resolution)):
            p.unlink(missing_ok=True)
    else:
        existing = _find_existing_tif(year, resolution)
        if existing is not None:
            return existing

    if not allow_download:
        expected = _data_root() / _tif_basename(year, resolution)
        raise FileNotFoundError(
            f"CDL raster missing: {expected}\n"
            f"Engine is in strict mode — no implicit downloads, no "
            f"~/.hack26 fallback.\n"
            f"Either pre-stage the file (workshop EFS should ship "
            f"2025_10m_cdls.tif), or call with allow_download=True "
            f"(CLI: --allow-download) to fetch into the data root."
        )

    zip_path = _download_zip(year, resolution, source=source)
    return _extract_tif(zip_path, year, resolution)


# ---------------------------------------------------------------------------
# Per-county aggregation
# ---------------------------------------------------------------------------

def _county_histogram(src, geometry, src_crs) -> tuple["np.ndarray", int]:  # type: ignore[name-defined]
    """Crop ``src`` to ``geometry`` (in EPSG:4269) and return the per-class
    pixel histogram (length 256) plus the raster pixel area in m²."""
    # Lazy imports so `from engine import cdl` works without rasterio installed
    # — useful for inspecting URL/cache constants on the local dev box.
    import numpy as np
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping

    geom_proj = transform_geom("EPSG:4269", src_crs, mapping(geometry))
    # crop=True trims the read window to the polygon's bbox (huge speedup on a
    # 14 GB national raster). filled=False gives us a masked array so pixels
    # outside the polygon are excluded from the histogram, not counted as 0.
    out_image, _ = rio_mask(
        src, [geom_proj], crop=True, filled=False, nodata=src.nodata,
    )
    band = out_image[0]
    # Combine the polygon mask with the nodata mask. CDL nodata is usually 0,
    # but rasterio also marks pixels outside the read mask. Both should be skipped.
    inside = ~np.ma.getmaskarray(band)
    values = np.asarray(band)[inside].astype(np.int64, copy=False)
    # CDL classes fit in a byte; bincount with minlength=256 makes the column
    # layout stable so we can vstack across counties later if we need to.
    hist = np.bincount(values, minlength=256)
    # Pixel area in m² — CDL CRS is CONUS Albers (units: meters), so this is a
    # straight multiply of the affine pixel sizes.
    px_area = abs(src.transform.a * src.transform.e)
    return hist, int(round(px_area))


def _row_from_histogram(
    hist, geoid: str, year: int, resolution: int, px_area_m2: int
) -> dict:
    """Project the 256-bin histogram down to the columns the model actually uses."""
    total_px = int(hist.sum() - hist[0])  # exclude nodata/background
    cropland_px = int(
        sum(int(c) for cls, c in enumerate(hist) if cls not in NONCROPLAND_CLASSES)
    )
    corn_px = int(hist[CDL_CORN_FOR_GRAIN])
    sweet_corn_px = int(hist[CDL_SWEET_CORN])
    pop_corn_px = int(hist[CDL_POP_ORN_CORN])
    soy_px = int(hist[CDL_SOYBEANS])
    return {
        "geoid": geoid,
        "year": year,
        "resolution_m": resolution,
        "pixel_area_m2": px_area_m2,
        "total_pixels": total_px,
        "cropland_pixels": cropland_px,
        "corn_pixels": corn_px,
        "sweet_corn_pixels": sweet_corn_px,
        "pop_orn_corn_pixels": pop_corn_px,
        "soybean_pixels": soy_px,
        "corn_area_m2": corn_px * px_area_m2,
        "soybean_area_m2": soy_px * px_area_m2,
        # `corn_pct_of_county` denominator includes water, forest, etc. so it's
        # comparable across counties of different cropland intensity.
        # `corn_pct_of_cropland` is the corn share of *cultivated* land, which
        # is the more useful number for yield-weighted aggregation.
        "corn_pct_of_county": (corn_px / total_px) if total_px else 0.0,
        "corn_pct_of_cropland": (corn_px / cropland_px) if cropland_px else 0.0,
    }


def fetch_county_cdl(
    geoid: str,
    geometry,
    year: int = 2025,
    resolution: int = 30,
    src=None,
) -> pd.DataFrame:
    """Per-county corn statistics for one polygon. Single-row DataFrame.

    Pass ``src`` (an open ``rasterio.DatasetReader``) when looping over many
    counties — opening the 14 GB national raster once and reusing it is ~100x
    faster than reopening per call.
    """
    _validate(year, resolution)
    import rasterio

    if src is None:
        tif = load_cdl(year=year, resolution=resolution)
        with rasterio.open(tif) as opened:
            hist, px_area = _county_histogram(opened, geometry, opened.crs)
    else:
        hist, px_area = _county_histogram(src, geometry, src.crs)

    return pd.DataFrame([_row_from_histogram(hist, geoid, year, resolution, px_area)])


def fetch_counties_cdl(
    counties,
    year: int = 2025,
    resolution: int = 30,
    refresh: bool = False,
    allow_download: bool = False,
) -> pd.DataFrame:
    """Vectorized version of :func:`fetch_county_cdl` over a county GeoDataFrame.

    Opens the national raster exactly once and walks each row. Returns one
    DataFrame keyed by ``geoid``. Result is cached as parquet under
    ``<data_root>/derived/county_features_{year}_{res}m_{nrows}_{hash}.parquet``
    (i.e. alongside the EFS rasters — never ``~/.hack26``) so a repeat call
    with the same county set is an in-memory read. The hash is a short
    digest over the sorted geoids so two different county sets of the same
    size (e.g. 5 Iowa counties vs 5 Colorado counties) get distinct cache
    files instead of silently colliding.
    """
    _validate(year, resolution)
    import rasterio

    # Content-addressed cache key: sorted geoids → SHA-1 → first 12 hex chars.
    # Sorted so row-order differences don't cause a false miss; truncated
    # because filename length matters more than collision resistance here.
    geoids_sorted = sorted(str(g) for g in counties["geoid"])
    geoid_hash = hashlib.sha1("\n".join(geoids_sorted).encode("utf-8")).hexdigest()[:12]
    cache = (
        _derived_dir()
        / f"county_features_{year}_{resolution}m_{len(counties)}_{geoid_hash}.parquet"
    )
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    tif = load_cdl(year=year, resolution=resolution, refresh=refresh,
                   allow_download=allow_download)

    rows: list[dict] = []
    with rasterio.open(tif) as src:
        src_crs = src.crs
        # Reproject all polygons to the raster CRS up front so the inner loop
        # only does the rio_mask call. This also gives us a single place to
        # report any CRS mismatch instead of one error per county.
        for i, (_, county) in enumerate(counties.iterrows(), start=1):
            geoid = str(county["geoid"])
            try:
                hist, px_area = _county_histogram(src, county.geometry, src_crs)
            except Exception as exc:  # noqa: BLE001 - we re-raise with context
                raise RuntimeError(
                    f"CDL aggregation failed for geoid={geoid} ({county.get('name_full', '?')}): {exc}"
                ) from exc
            rows.append(_row_from_histogram(hist, geoid, year, resolution, px_area))
            if i % 25 == 0 or i == len(counties):
                print(f"[cdl] {i}/{len(counties)} counties processed", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _summarize(df: pd.DataFrame, year: int, resolution: int, tif: Path) -> str:
    by_state = (
        df.assign(state_fips=df["geoid"].str[:2])
        .groupby("state_fips", as_index=False)
        .agg(
            n_counties=("geoid", "size"),
            corn_area_km2=("corn_area_m2", lambda s: round(s.sum() / 1e6, 1)),
            mean_corn_pct=("corn_pct_of_county", lambda s: round(s.mean() * 100, 2)),
        )
        .sort_values("state_fips")
    )
    return "\n".join(
        [
            f"year:           {year}",
            f"resolution:     {resolution} m",
            f"raster:         {tif}",
            f"counties:       {len(df)}",
            f"total corn:     {round(df['corn_area_m2'].sum() / 1e6, 1):,} km²",
            "",
            by_state.to_string(index=False),
        ]
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build/refresh per-county CDL features from the USDA national raster."
    )
    parser.add_argument("--year", type=int, default=LATEST_YEAR,
                        help=f"CDL year ({EARLIEST_YEAR}-{LATEST_YEAR}, default {LATEST_YEAR}).")
    parser.add_argument("--resolution", type=int, default=30, choices=(10, 30),
                        help="Pixel resolution in meters. 10 m only exists for 2024-2025.")
    parser.add_argument("--states", nargs="+", default=None, metavar="STATE",
                        help="Subset to one or more state names / 2-digit FIPS. "
                             "Omit for all 5 target states.")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-extract + re-aggregate. Implies --allow-download "
                             "for the raster step if the raster is missing.")
    parser.add_argument("--source", choices=("auto", "workshop", "nass"), default="auto",
                        help="Where to fetch the zip from when a download is needed. "
                             "Only consulted under --allow-download / --download-only.")
    parser.add_argument("--allow-download", action="store_true",
                        help="Permit fetching the national raster from NASS / workshop S3 "
                             "into the data root if it isn't already pre-staged. Off by "
                             "default — engine refuses implicit multi-GB downloads.")
    parser.add_argument("--download-only", action="store_true",
                        help="Just fetch + extract the national raster (writes into the "
                             "data root); skip per-county work. Implies --allow-download.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional output path (.parquet or .csv) for the per-county frame.")
    args = parser.parse_args(argv)

    try:
        _validate(args.year, args.resolution)
    except ValueError as e:
        parser.error(str(e))

    if args.download_only:
        tif = load_cdl(year=args.year, resolution=args.resolution,
                       refresh=args.refresh, source=args.source,
                       allow_download=True)
        print(f"raster ready: {tif}")
        return 0

    # Pre-flight: surface the strict-mode FileNotFoundError before we do any
    # geopandas work, so the operator gets a clean message instead of a
    # mid-pipeline traceback.
    load_cdl(year=args.year, resolution=args.resolution,
             refresh=args.refresh, source=args.source,
             allow_download=args.allow_download)

    # Lazy import so the module is usable for inspection without geopandas wired up.
    from engine.counties import load_counties

    counties = load_counties(states=args.states)
    df = fetch_counties_cdl(counties, year=args.year, resolution=args.resolution,
                            refresh=args.refresh)
    tif = _find_existing_tif(args.year, args.resolution) or Path("<missing>")
    print(_summarize(df, args.year, args.resolution, tif))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        suffix = args.out.suffix.lower()
        if suffix == ".parquet":
            df.to_parquet(args.out, index=False)
        elif suffix == ".csv":
            df.to_csv(args.out, index=False)
        else:
            raise SystemExit(f"unsupported --out suffix: {suffix} (use .parquet or .csv)")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
