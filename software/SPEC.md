# Geospatial AI Crop Yield Forecasting — System Spec

## 1. Problem

Forecast **corn-for-grain yield (bu/acre)** for **Iowa, Colorado, Wisconsin, Missouri, Nebraska** at four points in the growing season (Aug 1, Sep 1, Oct 1, final), each wrapped in an analog-year **cone of uncertainty**.

Replacement target: the USDA enumerator survey (~1,600 boots-on-the-ground, ~$1–1.5M per pass, 4×/year, with dwindling participation).

## 2. Architecture — "ROI in, forecast out"

Solid boxes are implemented; dashed boxes are planned components that plug into the same Engine contract.

```mermaid
flowchart TD
    User["User / CLI"] -->|state or geoid list| Catalog["County Catalog<br/><i>canonical IDs + geometry</i>"]
    Catalog -->|geoid, geometry| Engine

    subgraph Engine["Engine — pluggable data sources"]
        direction LR
        Counties["counties (built)"]
        CDL["CDL corn mask (built)"]
        HLS["Imagery — HLS"]:::planned
        WX["Weather<br/>POWER / ERA5 / GEFS"]:::planned
        SM["Soil moisture — SMAP"]:::planned
        NASS["NASS yields<br/>(ground truth)"]:::planned
    end

    Engine -->|feature frame<br/>ROI × season × as-of date| Model["Model + Analogs"]:::planned
    Model --> Forecast["Forecast<br/>per state × date"]:::planned

    classDef planned stroke-dasharray: 4 3, color:#888;
```

Design rules every Engine source follows:
- **Join key is `geoid`** (5-digit county FIPS).
- **Each source is a function** `fetch(geoid, geometry, date_range) -> pd.DataFrame`. Sources are independent, cacheable, and individually testable.
- **The County Catalog owns geometry.** Downstream sources receive the polygon; they never re-derive it.

## 3. Region of Interest (ROI)

MVP scope is **county-level** ROIs in the 5 target states (443 counties total: CO 64, IA 99, MO 115, NE 93, WI 72). County granularity matches USDA NASS's published corn yields, giving the richest training signal at a tractable scale.

The Engine contract takes a generic polygon, so any sub-county ROI (a producer's field, a watershed, an AgNext research plot) plugs in without code changes.

## 4. Component: County Catalog `engine.counties`

**Purpose.** Return one canonical `GeoDataFrame` of every county in the 5 target states, keyed by `geoid`, carrying the geometry every other Engine source needs.

**Source.** Census Bureau TIGER/Line **2024** national county shapefile — single authoritative file, free, no auth. Pinned vintage. Cached locally on first call.

**State FIPS in scope.**

| State     | FIPS |
| --------- | ---- |
| Colorado  | 08   |
| Iowa      | 19   |
| Missouri  | 29   |
| Nebraska  | 31   |
| Wisconsin | 55   |

**Output schema.**

| Column          | Type            | Notes                                              |
| --------------- | --------------- | -------------------------------------------------- |
| `geoid`         | str (5)         | Primary key. State FIPS + county FIPS.             |
| `state_fips`    | str (2)         |                                                    |
| `county_fips`   | str (3)         |                                                    |
| `name`          | str             | "Story", "Larimer", …                              |
| `name_full`     | str             | "Story County", "Larimer County", …                |
| `state_name`    | str             | Human-readable state.                              |
| `centroid_lat`  | float           | TIGER `INTPTLAT` (interior point, not bbox).       |
| `centroid_lon`  | float           | TIGER `INTPTLON`.                                  |
| `land_area_m2`  | Int64           | TIGER `ALAND`. For per-area normalization.         |
| `water_area_m2` | Int64           | TIGER `AWATER`.                                    |
| `geometry`      | shapely Polygon | EPSG:4269 (NAD83), as published by Census.         |

Invariants asserted before the frame is returned: `geoid` is unique, `geoid` is exactly 5 chars, every row has a non-null geometry.

**Public API.**
```python
from engine.counties import load_counties

gdf = load_counties()                       # all 5 states
gdf = load_counties(states=["Iowa"])        # subset by name or FIPS
gdf = load_counties(refresh=True)           # re-download + rebuild cache
```

`states=` accepts state names ("Iowa") or 2-digit FIPS ("19"); unknown values raise `ValueError`.

**Call flow.**

```mermaid
sequenceDiagram
    participant Caller as Caller (CLI / Engine)
    participant Catalog as load_counties()
    participant Cache as Local cache<br/>(~/.hack26/tiger/)
    participant TIGER as Census TIGER/Line 2024

    Caller->>Catalog: load_counties(states=[...])
    Catalog->>Cache: counties_5state_2024.parquet present?
    alt parquet hit
        Cache-->>Catalog: GeoParquet
    else parquet miss
        Catalog->>Cache: tl_2024_us_county.zip present?
        alt zip miss
            Catalog->>TIGER: GET tl_2024_us_county.zip
            TIGER-->>Catalog: zipped shapefile
            Catalog->>Cache: persist zip
        end
        Catalog->>Catalog: read shapefile, filter to 5 states,<br/>derive geoid, normalize columns, validate
        Catalog->>Cache: write GeoParquet
    end
    Catalog->>Catalog: filter to requested `states`
    Catalog-->>Caller: GeoDataFrame (schema above)
```

The cache has two layers: the raw TIGER zip (so a refresh can rebuild without re-downloading ~120 MB) and the normalized 5-state GeoParquet (so warm reads are sub-second). `refresh=True` rebuilds both.

**Cache location.** `~/.hack26/tiger/` by default. Override with the `HACK26_CACHE_DIR` environment variable (the `tiger/` subdir is appended automatically).

**Non-goals.**
- No reprojection — downstream sources reproject to whatever they need (HLS → UTM, NASS → FIPS-keyed only).
- No sub-county geometries.
- No alternate vintages — TIGER 2024 is pinned via `TIGER_YEAR` in `engine/counties.py`.

## 5. Component: CDL Corn Mask `engine.cdl`

**Purpose.** Project the USDA Cropland Data Layer national raster down to per-county corn statistics keyed on `geoid`, so it slots straight into the SPEC §2 Engine contract and joins against the County Catalog without any glue.

**Source.** USDA NASS Cropland Data Layer — annual, geo-referenced, crop-specific raster covering CONUS. Available years and resolutions:

| Resolution | Years     | Note                                                          |
| ---------- | --------- | ------------------------------------------------------------- |
| 30 m       | 2008–2025 | Resampled from the 10 m product for 2024+; native for ≤2023.  |
| 10 m       | 2024–2025 | Native generation; ~9.8 GB zipped, ~14.9 GB extracted (2025). |

Downloads go through one of two endpoints:
- **NASS HTTPS** — `https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{year}_{res}m_cdls.zip`. Default. Covers every (year, resolution) combination above.
- **Workshop S3 mirror** — `s3://rayette.guru/workshop/2025_10m_cdls.zip`, anonymous read via `aws s3 cp --no-sign-request`. Only hosts 2025 10 m. Used automatically when running on the AWS sagemaker box for that one combo (faster in-region transfer); otherwise NASS is used.

**Output schema** (one row per county, returned by `fetch_counties_cdl`):

| Column                  | Type    | Notes                                                              |
| ----------------------- | ------- | ------------------------------------------------------------------ |
| `geoid`                 | str (5) | Join key — matches the County Catalog.                             |
| `year`                  | int     | CDL vintage.                                                       |
| `resolution_m`          | int     | 10 or 30.                                                          |
| `pixel_area_m2`         | int     | `resolution_m ** 2`. Surfaced so callers don't have to recompute.  |
| `total_pixels`          | int     | All non-background pixels inside the county polygon.               |
| `cropland_pixels`       | int     | Excludes water, developed, forest, wetlands (CDL classes 63–64, 81–92, 111–195). |
| `corn_pixels`           | int     | CDL class 1 (corn-for-grain — the replacement target).             |
| `sweet_corn_pixels`     | int     | CDL class 12.                                                      |
| `pop_orn_corn_pixels`   | int     | CDL class 13.                                                      |
| `soybean_pixels`        | int     | CDL class 5. Surfaced because corn↔soy rotation is a strong predictor. |
| `corn_area_m2`          | int     | `corn_pixels * pixel_area_m2`.                                     |
| `soybean_area_m2`       | int     | `soybean_pixels * pixel_area_m2`.                                  |
| `corn_pct_of_county`    | float   | `corn_pixels / total_pixels`. Comparable across counties of different cropland intensity. |
| `corn_pct_of_cropland`  | float   | `corn_pixels / cropland_pixels`. Better for yield-weighted aggregation. |

**Public API.**
```python
from engine.cdl import load_cdl, fetch_county_cdl, fetch_counties_cdl
from engine.counties import load_counties

tif = load_cdl(year=2025, resolution=10)                   # Path to national GeoTIFF
df  = fetch_counties_cdl(load_counties(states=["Iowa"]),   # one row per county
                         year=2025, resolution=10)
row = fetch_county_cdl(geoid="19169", geometry=poly,       # single-county form
                       year=2024, resolution=30)
```

`load_cdl(year, resolution)` validates the (year, resolution) combo against the matrix above and raises `ValueError` for unsupported pairs (e.g. 2019 at 10 m).

**Data discovery.** `load_cdl` probes these locations *in order* before triggering a download. This is how the AWS sagemaker workshop machine reuses its pre-extracted 14.9 GB EFS copy instead of pulling another 9.8 GB:

1. `$HACK26_CDL_DATA_DIR/{year}_{res}m_cdls.tif` — explicit env override.
2. `~/hack26/data/{year}_{res}m_cdls.tif` — the documented EFS mount per the README.
3. `~/.hack26/cdl/{year}_{res}m_cdls.tif` — our own download cache.
4. Download zip → extract → step 3.

**Call flow.**

```mermaid
sequenceDiagram
    participant Caller as Caller (CLI / Engine)
    participant CDL as load_cdl()
    participant EFS as EFS mount<br/>(~/hack26/data)
    participant Cache as Local cache<br/>(~/.hack26/cdl/)
    participant S3 as Workshop S3<br/>(2025 10 m only)
    participant NASS as NASS HTTPS<br/>(2008-2025)

    Caller->>CDL: load_cdl(year, resolution)
    CDL->>EFS: {year}_{res}m_cdls.tif present?
    alt EFS hit
        EFS-->>Caller: Path (no I/O)
    else EFS miss
        CDL->>Cache: extracted .tif present?
        alt cache hit
            Cache-->>Caller: Path
        else cache miss
            CDL->>Cache: zip present?
            alt zip miss
                alt year=2025, res=10, aws CLI present
                    CDL->>S3: anonymous GET
                    S3-->>Cache: zip
                else
                    CDL->>NASS: GET
                    NASS-->>Cache: zip
                end
            end
            CDL->>Cache: extract .tif (+ .ovr / .aux sidecars)
            Cache-->>Caller: Path
        end
    end
```

`fetch_counties_cdl` opens the national raster once, reprojects each county polygon from EPSG:4269 (NAD83) to the CDL's CONUS Albers CRS via `rasterio.warp.transform_geom`, and runs `rasterio.mask` per county to get a 256-bin pixel-class histogram. The output frame is cached as `~/.hack26/cdl/county_features_{year}_{res}m_{nrows}_{geoid_hash}.parquet` (the hash is a 12-char SHA-1 prefix over the sorted `geoid` list, so different county sets of the same size — e.g. 5 Iowa counties vs 5 Colorado counties — never share a cache file) so a repeat call with the same county set is a sub-second parquet read.

**Cache layout.**
```
~/.hack26/cdl/
├── 2025_10m_cdls.zip                   # raw NASS / workshop download (only if EFS missed)
├── 2025_10m_cdls.tif                   # extracted national raster
├── 2025_10m_cdls.tif.ovr               # overview pyramid sidecar
└── county_features_2025_10m_99_3f7a9c1b2e4d.parquet # per-county aggregation (Iowa example; suffix is sha1(sorted geoids)[:12])
```

`refresh=True` only ever clobbers files in `~/.hack26/cdl/`. Pre-mounted EFS rasters at `~/hack26/data/` are treated read-only.

**Non-goals.**
- No raster reprojection — we always warp the county polygon to CDL Albers, never the other way around (a national 10 m reproject would be a tens-of-GB operation per call).
- No sub-county aggregation — `fetch_county_cdl` accepts an arbitrary polygon, so field-level callers plug in via the same function; the per-state CLI just wraps the county geometry case.
- No confidence-layer ingest — NASS publishes a separate `{year}_30m_Confidence_Layer.zip`; out of scope for the MVP.

## 6. Repository layout

```
hack26/
├── pyproject.toml           # source of truth for deps, package config, pytest
├── software/
│   ├── requirements.txt     # locked runtime deps (uv pip compile output)
│   ├── requirements-dev.txt # locked runtime + dev deps
│   ├── engine/
│   │   ├── __init__.py      # lazy re-exports for all built sources
│   │   ├── counties.py      # County Catalog implementation + CLI
│   │   └── cdl.py           # CDL Corn Mask implementation + CLI
│   └── tests/
│       ├── test_counties_smoke.py
│       └── test_cdl_smoke.py
└── .venv/                   # local environment (gitignored)
```

Local caches (gitignored, live outside the repo):
- `~/.hack26/tiger/` — TIGER zip + 5-state GeoParquet.
- `~/.hack26/cdl/` — CDL zips, extracted GeoTIFFs, per-county feature parquets.
- `~/hack26/data/` — read-only EFS mount on the AWS workshop machine; pre-extracted national CDL rasters live here.

## 7. Operations

### 7.1 First-time environment setup

Requires Python ≥ 3.11. `uv` is recommended for speed but optional.

```powershell
# Option A — uv (fast)
python -m uv venv .venv --python 3.13
python -m uv pip install --python .venv\Scripts\python.exe -e ".[dev]"

# Option B — stock pip + venv
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Cloud worker (no editable install needed if you only want to run, not modify):
```bash
pip install -r software/requirements.txt
pip install -e .   # registers the `engine` package
```

### 7.2 Running the County Catalog

Module form:
```powershell
.venv\Scripts\python.exe -m engine.counties                       # download + cache, print summary
.venv\Scripts\python.exe -m engine.counties --refresh             # force re-download + rebuild
.venv\Scripts\python.exe -m engine.counties --states Iowa Colorado
.venv\Scripts\python.exe -m engine.counties --out catalog.parquet # also export a copy (.parquet or .csv)
```

Console-script form (after editable install):
```powershell
.venv\Scripts\hack26-counties.exe --states Iowa
```

### 7.3 Running the CDL Corn Mask

Intended to run on the AWS sagemaker workshop instance — the 14.9 GB 2025 10 m raster is already on the EFS mount at `~/hack26/data/`, so the first call only does per-county masking, not a download.

Module form:
```bash
python -m engine.cdl                                    # 2025, 30 m, all 5 states
python -m engine.cdl --year 2024 --resolution 30        # any historical year (2008-2025 @ 30 m)
python -m engine.cdl --year 2025 --resolution 10        # uses EFS .tif if present, else downloads
python -m engine.cdl --states Iowa Colorado             # subset
python -m engine.cdl --refresh                          # re-download zip + re-aggregate
python -m engine.cdl --download-only                    # fetch + extract only, skip masking
python -m engine.cdl --source nass                      # force NASS HTTPS (default is auto)
python -m engine.cdl --source workshop                  # force s3://rayette.guru (2025 10 m only)
python -m engine.cdl --out cdl_features.parquet         # also export (.parquet or .csv)
```

Console-script form (after editable install):
```bash
hack26-cdl --year 2025 --resolution 10 --states Iowa
```

Expected runtime on the AWS box (2025 10 m, all 5 states / 443 counties): ~minutes for the first cold pass; sub-second for cached re-reads (parquet at `~/.hack26/cdl/county_features_2025_10m_443_<hash>.parquet`).

### 7.4 Tests

```powershell
.venv\Scripts\python.exe -m pytest software\tests -v
```

Two smoke tests:
- `software/tests/test_counties_smoke.py` — loads Colorado, eyeballs the first 5 counties, asserts schema + filter + geometry validity + centroid-in-CO-bbox. First run ~10 s (TIGER download); cached <1 s.
- `software/tests/test_cdl_smoke.py` — runs `fetch_counties_cdl` against the first 5 Iowa counties and asserts schema + per-county corn fraction is in a sane Iowa range (5–85 %). **Auto-skips** when rasterio isn't installed or no national CDL `.tif` is reachable on disk, so it never triggers a multi-GB download from a CI pipe.

Standalone form (each test file is also runnable with `python -m`):
```powershell
.venv\Scripts\python.exe software\tests\test_counties_smoke.py
.venv\Scripts\python.exe software\tests\test_cdl_smoke.py
```

### 7.5 Refreshing the dependency lock

When `pyproject.toml`'s `[project.dependencies]` or `[project.optional-dependencies]` change (e.g. when `rasterio` was added for the CDL source):

```powershell
python -m uv pip compile pyproject.toml -o software\requirements.txt
python -m uv pip compile pyproject.toml --extra dev -o software\requirements-dev.txt
```

### 7.6 Refreshing source caches

TIGER (county catalog):
```powershell
.venv\Scripts\python.exe -m engine.counties --refresh
```

CDL (only files in `~/.hack26/cdl/` — pre-mounted EFS rasters at `~/hack26/data/` are read-only and ignored):
```bash
python -m engine.cdl --year 2025 --resolution 30 --refresh
```

Or, to start fully clean:
```powershell
Remove-Item -Recurse -Force $HOME\.hack26
```

### 7.7 Environment variables

| Var                  | Default              | Effect                                                        |
| -------------------- | -------------------- | ------------------------------------------------------------- |
| `HACK26_CACHE_DIR`   | `~/.hack26`          | Root for all source caches. `tiger/` and `cdl/` subdirs created underneath. |
| `HACK26_CDL_DATA_DIR`| *(unset)*            | First location probed for an already-extracted CDL `{year}_{res}m_cdls.tif`. Use this on the AWS box to point at a non-default EFS mount. Falls back to `~/hack26/data/` then `$HACK26_CACHE_DIR/cdl/`. |
