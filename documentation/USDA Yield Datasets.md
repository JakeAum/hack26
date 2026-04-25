# USDA Crop Yield Datasets — Research Notes

Goal: figure out the **standard format** the USDA publishes corn yield data in, so the model we eventually train can output to the same shape (units, granularity, and reporting cadence). The headline question — "are historic yields and forecasts attributed at the county or state level?" — is answered first, then per-source detail follows.

## TL;DR — granularity answer

| Layer | County-level? | State-level? | Notes |
| --- | --- | --- | --- |
| **Final / actual** corn-for-grain yield (bu/acre), end of season | ✅ **Yes** (~1,300+ counties / yr nationwide; ~all 443 of our 5 states qualify, see below) | ✅ Yes | Counties published in **late May** the year *after* harvest. State final published in **early January**. |
| **In-season forecasts** (the 4 dates we care about: Aug 1 / Sep 1 / Oct 1 / final) | ❌ **No.** NASS does not publish county-level **in-season** forecasts. | ✅ Yes — official NASS forecasts are **state and national only**. | This is exactly the gap our pipeline targets. The 4-times-a-year USDA forecast cycle = May / Aug / Sep / Oct / Nov *Crop Production* reports, all state+national. |
| Cropland area / corn pixel coverage (CDL-derived) | ✅ Yes (we already do this in `engine.cdl`) | ✅ Yes | Not a yield, but joins on `geoid`. |
| Soil moisture / NDVI / VCI (Crop-CASMA) | ✅ Yes — county zonal stats are a first-class export | ✅ Yes | Pixel raster, downloadable as CSV per AOI. |

**Practical implication for our model output schema:**

- **Ground truth for training** → county-level NASS yields (`agg_level_desc=COUNTY`, `short_desc="CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"`, `reference_period_desc="YEAR"`). One row per `(geoid, year)`.
- **Replacement target / what we output** → match the *USDA forecast format* but at finer granularity. USDA publishes state forecasts on **Aug / Sep / Oct / Nov** *Crop Production* reports (`reference_period_desc="YEAR - AUG FORECAST"` etc.) → we can output **both** the state row (so it's a drop-in replacement for the USDA enumerator survey) and the county row (the value-add we get from going pixel-level). Our SPEC §1 already targets exactly this cadence: "Aug 1, Sep 1, Oct 1, final".

So the canonical output row for our model should be a superset of NASS Quick Stats:

```
geoid (5-digit FIPS, "00<state_fips>" for state row)
agg_level_desc        ∈ {STATE, COUNTY}
state_fips, state_alpha, state_name
county_ansi, county_name (null when STATE)
year
reference_period_desc ∈ {YEAR - AUG FORECAST, YEAR - SEP FORECAST,
                         YEAR - OCT FORECAST, YEAR}     # YEAR = final
load_time             # forecast date (the "as-of")
yield_bu_per_acre     # the prediction
yield_lo_bu_per_acre  # cone-of-uncertainty lower bound
yield_hi_bu_per_acre  # cone-of-uncertainty upper bound
```

That mapping makes our forecasts trivially comparable to NASS's published numbers via the Quick Stats schema (more on the schema below).

---

## 1) USDA NASS Quick Stats — the canonical yield store

The chart at `https://www.nass.usda.gov/Charts_and_Maps/graphics/cornyld.pdf` is just a rendering of one Quick Stats query — annual U.S. corn-for-grain yield over time. The underlying database is **Quick Stats** at `https://quickstats.nass.usda.gov/` with a free API at `https://quickstats.nass.usda.gov/api`.

### Granularity

The schema field that decides granularity is **`agg_level_desc`** (Quick Stats column "Geographic Level"). Possible values include:

- `NATIONAL`
- `STATE`
- `AGRICULTURAL DISTRICT` (NASS-defined county groupings, ~9 per state, identified by `asd_code` / `asd_desc`)
- `COUNTY` (identified by `state_ansi` + `county_ansi` → 5-digit FIPS, **same join key as our `geoid`**)
- `WATERSHED`, `ZIP CODE`, `REGION`, `CONGRESSIONAL DISTRICT` (less relevant to us)

### County-level corn yield (the ground truth column for training)

- Series name: `CORN, GRAIN - YIELD, MEASURED IN BU / ACRE`
- Filter: `commodity_desc=CORN`, `class_desc=GRAIN`, `statisticcat_desc=YIELD`, `unit_desc=BU / ACRE`, `agg_level_desc=COUNTY`
- Granularity: **per-county per-year** ("County estimates are the only source of yearly localized estimates" — NASS)
- Coverage: corn-for-grain county estimates are produced for **31 states**, including all 5 in our scope (CO, IA, MO, NE, WI). In 2024, 1,296 counties were published with a non-aggregated number; the rest fall into a per-state catch-all bucket called "OTHER (COMBINED) COUNTIES" with `county_code=998`. Counties get suppressed when reporting standards aren't met (too few respondents or yielded acres) — those counties don't disappear, they're rolled into the "Other" bucket so state totals still reconcile.
- Release timing: **end of May the year after harvest** (e.g. 2025 county yields publish ~May 2026). State-level finals come out ~5 months earlier (early January).

### How the county number is actually produced (post-2020)

This was a key surprise — **county-level yields are themselves model outputs, not direct survey averages**:

> Beginning with the 2020 crop year, model-based county-level estimates were incorporated into the NASS estimation process for row crops county estimates. Bayesian small area estimation models for county-level planted acreage, harvested acreage, and **yield** are fit separately for each crop … Yield models input current survey ratios, survey standard errors, and the **National Commodity Crop Productivity Index (NCCPI)**. … These county-level estimates are **benchmarked against previously released state-level official estimates** established by the Agricultural Statistics Board (ASB).

— *Row Crops County Estimates Methodology and Quality Measures*, NASS, June 2025

Two consequences for us:

1. The published county "actual" we use as ground truth is itself a posterior mean from a Bayesian small-area model conditioned on the state-level total. If we ever want a noise-floor / irreducible-error baseline, NASS publishes per-county **CVs** (coefficient of variation) — for 2024 the median yield CV for corn-for-grain was **6.5%** (25th=3.6%, 75th=9.4%). That's the bar to beat.
2. Because counties are benchmarked to add up to state, our model's county forecasts should ideally do the same — sum (county forecast × harvested acres) → state total. Worth bookkeeping in the output.

### In-season forecasts (the four dates in our SPEC)

NASS publishes the in-season corn yield forecasts in monthly *Crop Production* reports keyed by `reference_period_desc`:

| `reference_period_desc` | Cycle | Geographic level published |
| --- | --- | --- |
| `YEAR - AUG FORECAST` | Aug *Crop Production* (≈ Aug 12) | National + State |
| `YEAR - SEP FORECAST` | Sep *Crop Production* (≈ Sep 12) | National + State |
| `YEAR - OCT FORECAST` | Oct *Crop Production* (≈ Oct 12) | National + State |
| `YEAR - NOV FORECAST` | Nov *Crop Production* (≈ Nov 9) | National + State |
| `YEAR` (final) | Jan *Crop Production Annual* (≈ Jan 12 next yr) | National + State (final) → County (~end May) |

This is exactly the SPEC §1 cadence (Aug 1, Sep 1, Oct 1, final). USDA's forecast is *always* state-only — the county-level row in Quick Stats only ever has the final `YEAR` value, no intermediate forecasts. **That's the gap.**

The in-season forecasts come from two big surveys, both subsampled from the June Area Survey:

- **Agricultural Yield (AY) Survey** — grower-reported, monthly Aug–Nov. About 70k farms.
- **Objective Yield (OY) Survey** — physical field counts (ear count, kernel rows, kernel weight) for corn, soybeans, cotton, winter wheat. Several thousand fields, in-person counting plots.

NASS bakes weather "departures from normal" into the regression but explicitly does **not** use any forward weather forecasts — every release is conditioned on conditions as of the first of the month. This is a useful design constraint to mirror in our pipeline so our outputs are defensible against the same baseline.

### Quick Stats API access

- Endpoint: `https://quickstats.nass.usda.gov/api/api_GET/?key=<KEY>&...`
- Auth: free API key (email-based registration).
- Output formats: **JSON / CSV / XML** (`format=` param).
- Limit: 50,000 rows per call → use `get_counts` first, paginate by year if needed. For our 5 states × 443 counties × ~25 years × 4 forecast dates we're well under this; a single call per state per year batch suffices.
- Existing Python wrapper: `nassqs` (PyPI, MIT). Mirrors the R `rnassqs` package referenced above. Worth pulling in.

A minimal county-yield pull for our scope looks like:

```
GET https://quickstats.nass.usda.gov/api/api_GET/
    ?key=<KEY>
    &source_desc=SURVEY
    &commodity_desc=CORN
    &class_desc=GRAIN
    &statisticcat_desc=YIELD
    &unit_desc=BU / ACRE
    &agg_level_desc=COUNTY
    &state_alpha=IA       # repeat for CO, MO, NE, WI
    &year__GE=2000
    &format=CSV
```

Output columns we care about: `state_ansi`, `county_ansi` (→ build `geoid = state_ansi + county_ansi`), `year`, `reference_period_desc`, `Value` (string, has thousands commas — strip), `load_time` (the forecast/release date), `CV (%)` is on a sibling endpoint (`statisticcat_desc=CV (%)` with the rest the same).

This becomes the next planned engine module under `software/engine/nass/` — same `fetch(geoid, geometry, date_range)` contract as our other sources, except `geometry` is unused for NASS.

---

## 2) CroplandCROS

CroplandCROS is the **interactive viewer for the Cropland Data Layer (CDL)** — same raster we already ingest in `engine.cdl`. The "service" referred to in the hackathon brief is the public CDL web service backing it, which is *not* a yield product. It's a per-pixel crop-class label (e.g. "corn=1, soybeans=5") at 30 m or 10 m, plus zonal-statistics endpoints.

### What you can actually export

The viewer at `https://croplandcros.scinet.usda.gov/` exports:

- **Map images** (PDF / PNG / JPG) via the ArcGIS REST `Export Web Map` task at `https://pdi.scinet.usda.gov/hosting/rest/services/CroplandCROS_Print_Services/GPServer/Export Web Map` — visual only, not data.
- **Raw CDL raster** for a chosen AOI — useful but very large. We already get this via the national 10 m / 30 m TIFF in `engine.cdl`.
- **Zonal histograms / pixel-class statistics for an AOI** via the underlying CDL web service: `GetCDLStat` at `http://nassgeodata.gmu.edu:8080/axis2/services/CDLService/GetCDLStat?year=YYYY&fips=<5digit>&format=csv` — returns per-class pixel/area counts for that county. This is the same computation our `fetch_county_cdl` does locally.

### Granularity

CDL is **30 m or 10 m raster** — pixel-level, so any granularity ≥ pixel is supported. The viewer ships with built-in admin boundaries for **state, ag district, and county** AOIs, so county-level zonal stats are first-class. There is **no yield value** anywhere in CroplandCROS — only land-cover classification.

### Where CroplandCROS fits in our pipeline

It doesn't, beyond what we already do. It's the visualization layer for CDL, and our `engine.cdl` already does county zonal stats more efficiently than the per-county HTTP roundtrip would. The only reason to touch CroplandCROS would be:

1. **Visual demos** — embed the print-service map in a UI to show the corn mask we used.
2. **Sub-county AOIs we don't pre-stage** — the `GetCDLStat` web service is the easiest way to get pixel stats for an arbitrary polygon without downloading the 14.9 GB national TIFF, useful from a low-resource environment.

For training, treat CroplandCROS as an out-of-band artifact and keep using `engine.cdl`.

---

## 3) Crop-CASMA

Crop-CASMA = **Crop Condition and Soil Moisture Analytics**. NASA SMAP soil moisture + MODIS NDVI/VCI/MVCI, all rasterized and surfaced through a NASS-hosted viewer at `https://www.drought.gov/data-maps-tools/crop-condition-and-soil-moisture-analytics-tool-crop-casma`. User's guide is the source of truth: `https://nassgeo.csiss.gmu.edu/Crop-CASMA-User/`.

### What's available

| Layer family | Resolution | Cadence | Period |
| --- | --- | --- | --- |
| **NDVI** (vegetation greenness) | MODIS native (~500 m, sub-county) | Daily + weekly | 2000–present |
| **VCI** (Vegetation Condition Index — anomaly form of NDVI) | MODIS | Weekly | 2000–present |
| **MVCI** (Mean VCI, smoothed) | MODIS | Weekly | 2000–present |
| **SMAP surface soil moisture** | 9 km native (also a 1 km downscaled product) | Daily, weekly, **anomaly** | 2015–present (3-day lag on the public viewer) |
| **SMAP root-zone soil moisture** | 9 km | Daily, weekly, anomaly | 2015–present |

### Granularity & exports

The viewer's "zonal statistics" function lets you pick an AOI from built-in **state / county / agricultural district / CRD** boundaries (using Census admin polygons) **or** upload an arbitrary shapefile. For each AOI it returns per-pixel-class counts and acreage, plot-ready, with two main download paths:

- **Tabular zonal stats** → **CSV** (this is the one we'd use for training features)
- **Vector AOI** → **shapefile / GeoJSON / KML / GML** (reuse boundaries)
- **Raster clip** → GeoTIFF (the underlying SMAP/MODIS pixels for the AOI)

So everything Crop-CASMA exports is **at least county-resolvable**, and SMAP / NDVI products are sub-county (down to ~500 m–9 km depending on layer).

### Crop-CASMA vs what we already pull in `engine.weather`

Our existing `engine.weather` already gets the same physical signals from different APIs:

- NDVI / NDWI from **Sentinel-2 L2A** via Microsoft Planetary Computer (10–30 m, ~5-day revisit) → finer-resolution, slower to fetch.
- Surface soil moisture from **NASA POWER's `SMLAND`** column (POWER repackages SMAP) → already in our merged frame as `SMAP_surface_sm_m3m3`.

**Why Crop-CASMA still matters:**

1. **VCI / MVCI as anomaly products** — NASS already does the "departure from normal" math we'd otherwise have to derive from raw NDVI. These are the exact features the published yield models use (it's the same agency).
2. **SMAP root-zone + 1 km downscaled SMAP** — we currently only have surface SMAP. Root-zone is the more agronomically relevant layer for yield, and the 1 km downscaled product matches county-corn-pixel granularity better than 9 km.
3. **Authoritative provenance** — outputs on the same NDVI/SMAP grid NASS uses internally → easier defensibility against the official forecast.

### Access pattern

The viewer is point-and-click. For programmatic use there's a separate developer guide (`Crop-CASMA Developer's Guide`, `https://nassgeo.csiss.gmu.edu/Crop-CASMA-Developer/`) exposing the same layers via OGC WPS calls — workable but heavier than what we'd want to bake into the engine for the MVP. For now, treat Crop-CASMA as a manual export source for VCI/MVCI ground-truth-of-anomalies, and leave the SMAP/NDVI ingest path as it stands in `engine.weather`.

---

## 4) Things we should encode into the pipeline

Concretely, three follow-ups fall out of this research:

1. **Add an `engine.nass` module** following the existing `fetch(geoid, geometry, date_range) -> pd.DataFrame` contract. It pulls:
   - `agg_level_desc=COUNTY`, `commodity_desc=CORN`, `class_desc=GRAIN`, `statisticcat_desc=YIELD`, `unit_desc=BU / ACRE` → county final yields, ground truth.
   - The same query at `agg_level_desc=STATE` for the four `reference_period_desc` forecast values → the **baseline** to beat / the format we're emitting.
   - The CV column on a sibling call → per-county model uncertainty floor.
2. **Pin the model output schema** to the Quick Stats column set above (`agg_level_desc`, `geoid`, `year`, `reference_period_desc`, `Value`, `load_time`) plus our `yield_lo` / `yield_hi` cone bounds. That way our forecast CSV literally drops into Quick Stats-style downstream tooling (R `rnassqs`, Python `nassqs`, etc.).
3. **Document the "OTHER (COMBINED) COUNTIES" gotcha** — `county_code=998` rows are aggregates of suppressed counties, **not** a real county. Filter them out of training data before the join, or you'll double-count.

These map cleanly onto the SPEC §2 architecture (`NASS yields` is already a planned dashed box in the Engine subgraph) — the only addition is splitting that one box into "actuals (county)" vs "forecast baseline (state)" so the training data wiring and the evaluation wiring don't get confused.

---

## Sources

- NASS Quick Stats API docs: https://quickstats.nass.usda.gov/api
- NASS county estimates methodology (2025): https://www.nass.usda.gov/Publications/Methodology_and_Data_Quality/County_Estimates_Row_Crops/06_2025/rcceqm25.pdf
- NASS yield forecasting program (SEDMB 23-01): https://www.nass.usda.gov/Publications/Methodology_and_Data_Quality/Advanced_Topics/Yield%20Forecasting%20Program%20of%20NASS_2023.pdf
- CDL web service docs: https://www.nass.usda.gov/Research_and_Science/Cropland/docs/WebService.html
- CDL metadata: https://data.nass.usda.gov/Research_and_Science/Cropland/metadata/meta.php
- CroplandCROS viewer: https://croplandcros.scinet.usda.gov/
- Crop-CASMA viewer (NIDIS landing): https://www.drought.gov/data-maps-tools/crop-condition-and-soil-moisture-analytics-tool-crop-casma
- Crop-CASMA user's guide: https://nassgeo.csiss.gmu.edu/Crop-CASMA-User/
- Crop-CASMA metadata: https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_cropcasma.htm
