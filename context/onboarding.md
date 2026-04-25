# Onboarding Plan — CSU Geospatial AI Crop Yield Hackathon

A minimal, sequenced checklist to get from zero → "able to run a Prithvi notebook on real corn-belt imagery."

Target deliverable (reminder): corn-grain yield estimates for **IA, CO, WI, MO, NE** at **Aug 1, Sep 1, Oct 1, and end-of-season**, plus a **cone of uncertainty** built from analog weather years.

---

## Phase 0 — Accounts to create (do first, some take hours to approve)

| Account | Why | Link |
|---|---|---|
| NASA Earthdata Login | Required to pull HLS, MODIS, weather products | https://urs.earthdata.nasa.gov/users/new |
| Hugging Face | Download the Prithvi model weights | https://huggingface.co/join |
| USDA NASS QuickStats API key | Programmatic pull of corn yield ground truth | https://quickstats.nass.usda.gov/api |
| (optional) Google Earth Engine | Easiest way to pre-stage HLS/CDL/NAIP tiles | https://earthengine.google.com/signup/ |
| (optional) AWS account | HLS is mirrored on `s3://lp-prod-protected/HLSL30.020/` (free, requester-pays off) | https://aws.amazon.com/ |

After signup: accept the **Prithvi-EO-2.0** model license on Hugging Face — `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`.

---

## Phase 1 — Local environment

```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Create `requirements.txt` with the core stack and install:

```text
torch
torchvision
transformers
huggingface_hub
terratorch              # IBM/NASA's wrapper around Prithvi for fine-tuning
rasterio
rioxarray
xarray
geopandas
shapely
pyproj
earthengine-api        # optional, only if using GEE
planetary-computer     # optional, alternative HLS source
pystac-client
numpy
pandas
scikit-learn
matplotlib
jupyterlab
tqdm
python-dotenv
nasspython             # thin wrapper for USDA NASS QuickStats
```

```powershell
pip install -r requirements.txt
```

Add a `.env` (and `.gitignore` it):

```
EARTHDATA_USER=...
EARTHDATA_PASS=...
HF_TOKEN=...
NASS_API_KEY=...
```

---

## Phase 2 — Read / skim (≈2 hours total)

1. Prithvi-EO-2.0 model card — input bands, patch size, fine-tuning recipes.
   https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
2. TerraTorch quick-start — the official "fine-tune Prithvi on your raster + labels" framework.
   https://github.com/IBM/terratorch
3. HLS (Harmonized Landsat–Sentinel) product guide — bands, tiling, cadence.
   https://lpdaac.usgs.gov/products/hlsl30v002/
4. USDA NASS Cropland Data Layer (CDL) — corn class is `1`. Use to mask non-corn pixels.
   https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php
5. `context/agnext.md` (already in repo) — judge framing. Two of four judges are livestock/ag-science, **not CS**. Pitch must land for them.

---

## Phase 3 — Pull a tiny vertical slice (proves the pipeline end-to-end)

Pick **one county × one date** before scaling. Suggested: **Story County, Iowa, Aug 1 2023**.

1. **Yield label**: NASS QuickStats → `commodity_desc=CORN`, `statisticcat_desc=YIELD`, `unit_desc=BU / ACRE`, `agg_level_desc=COUNTY`, `state_alpha=IA`, `year=2023`.
2. **Crop mask**: download CDL 2023 for IA, reclassify (corn=1, else=0), clip to county.
3. **Imagery**: pull HLS L30/S30 tile(s) covering Story County for late July 2023, cloud-mask, compute NDVI/EVI as a sanity check.
4. **NAIP**: grab one NAIP tile from the same area for high-res context (used later for field boundaries).
5. **Weather**: pull NOAA gridMET or PRISM monthly Tmax/Tmin/precip/VPD for IA 2023.
6. **Model smoke test**: load Prithvi-EO-2.0 via TerraTorch, run a forward pass on the HLS chip, confirm you get embeddings out.

Save intermediate artifacts to `data/raw/`, `data/processed/`, `data/labels/` (gitignored).

---

## Phase 4 — Repo skeleton

Create this layout so work parallelizes cleanly:

```
hack26/
├─ data/                 # gitignored
│  ├─ raw/
│  ├─ processed/
│  └─ labels/
├─ notebooks/
│  ├─ 01_pull_nass.ipynb
│  ├─ 02_pull_hls.ipynb
│  ├─ 03_cdl_mask.ipynb
│  ├─ 04_prithvi_smoke_test.ipynb
│  └─ 05_yield_head_train.ipynb
├─ src/
│  ├─ data/              # downloaders + cleaners per source
│  ├─ features/          # NDVI/EVI/weather aggregations
│  ├─ models/            # Prithvi backbone + regression head
│  ├─ uncertainty/       # analog-year selection + CI
│  └─ pipelines/         # end-to-end orchestration
├─ context/              # already exists
├─ requirements.txt
├─ .env                  # gitignored
└─ README.md
```

---

## Phase 5 — Roadmap to deliverable (after smoke test passes)

1. **Scale data pull**: 5 states × 4 dates × 2005-2024 (training) + 2025 (inference). Probably the biggest time sink — start as soon as Phase 3 works.
2. **Train regression head** on Prithvi embeddings + weather features → bu/acre at county level, then aggregate to state.
3. **Cone of uncertainty**: for each (state, date), find K nearest historical seasons in weather-feature space → use the spread of NASS-reported yields from those analog years as the interval. Simple, defensible, and matches the problem statement.
4. **Inference for 2025** at the four dates.
5. **Presentation (5-7 min)**: data pipeline → model → uncertainty → results map. Frame impact in producer/policy terms (per `context/agnext.md`), not CS jargon.

---

## Definition of "onboarded"

You can check this list off and you're ready to build:

- [ ] Earthdata, Hugging Face, NASS API key all working from the CLI
- [ ] Prithvi license accepted, weights downloaded locally
- [ ] `.venv` activates and `import terratorch, rasterio, geopandas` succeeds
- [ ] One HLS tile + one CDL tile + one NASS yield row pulled for one county
- [ ] Prithvi forward pass produces embeddings on that HLS tile
- [ ] Repo skeleton committed
