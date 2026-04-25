# "We did not overfit" — chart pack for the demo deck

**Branch:** `research/robustness-and-metrics`  
**Date:** 2026-04-25  
**Companion:** `plot_training_metrics.py` (runnable counterpart of this doc)

The judges will ask one of three things:

1. *"Did the model just memorize the training years?"* → Charts 1, 2, 3, 7.
2. *"Are your uncertainty bands honest?"* → Charts 4, 5.
3. *"Does the model work where it wasn't trained?"* → Charts 6, 8.

This doc gives every chart a one-line claim, a passing-result threshold, and
the recipe. The companion script `plot_training_metrics.py` produces all of
them from `data/derived/logs/train_<variant>_<ts>.csv` plus
`data/derived/reports/test_<year>_predictions_<variant>.parquet` and the
NASS labels in the bundle's `series_index`.

---

## What the engine already gives us for free

Useful — minimal new instrumentation needed:

- **Per-epoch CSV** (`engine.model._make_csv_epoch_logger`) writes
  `ts, epoch, train_loss, val_loss, lr, elapsed_s, vram_alloc_gb, vram_peak_gb, tag`
  one row per validation epoch. This is the source for charts 1, 2, 3, 7.
- **Test predictions parquet** (`_train_one_pass` writes
  `reports/test_<year>_predictions_<variant>.parquet`) carries
  `geoid, year, forecast_date, yield_p10, yield_p50, yield_p90, yield_mean, yield_std`.
  Source for charts 4, 5, 6, 8.
- **Test metrics CSV** (`reports/test_<year>_metrics.csv`) carries per-state
  `n, rmse_bu_acre, mape_pct, p10_p90_coverage, label_mean, p50_mean`.
  Used as overlays in charts 4, 6.

Nice-to-haves we'd add later (each is a small edit to `engine.model`, none
required to ship the chart pack from RUN-4 data):

- Log per-quantile val loss separately (so chart 5 can show calibration
  *during training*, not only on test 2024).
- Log variable-selection weights at epoch end (so chart 7 can show the model
  isn't latched onto one feature). Darts exposes
  `model.model.encoder_variable_selection.softmax_weights` after fit.

---

## Chart 1 — Loss curves with best-epoch marker

**Claim:** training and validation losses both decrease, validation never
crashes through training (which would be a leak), and the best epoch is
clearly identified — we did not stop early or late by accident.

**Pass thresholds:**

- `train_loss` is monotonically (or near-monotonically) decreasing.
- `val_loss` reaches a global minimum and then plateaus or rises.
- The best epoch (annotated) is *not* the last epoch (which would mean we
  hadn't found the plateau yet).
- The final-epoch `val_loss` is **above** the final-epoch `train_loss`.
  *This is the visual you point at to refute "you must have leaked again."*

**X / Y:** epoch / loss (z-scored quantile loss).

**Layers:**

- Solid line train_loss (color A).
- Solid line val_loss (color B).
- Vertical dashed line at `argmin(val_loss)` labeled
  `best epoch = N`.
- Annotation block in the corner: best `val_loss`, final `train/val`,
  ratio.

**Source:** per-epoch CSV.

**Implementation:** `plot_training_metrics.py::plot_loss_curves`.

---

## Chart 2 — Generalization gap over time

**Claim:** the gap `val_loss − train_loss` is **bounded**. A growing gap
means the model is increasingly memorizing; a stable gap means it isn't.

**Pass thresholds:**

- After the initial transient (first ~5 epochs), the gap is roughly flat or
  trending downward.
- The best-epoch gap is < 0.30 in z-scored quantile-loss units (RUN-3 sat
  at ~0.10 at the best epoch and grew to ~0.27 by the EarlyStopping point).

**X / Y:** epoch / `val_loss − train_loss`.

**Layers:**

- Solid line for the gap.
- Horizontal reference at 0.
- Shaded band for "healthy" (< 0.20) vs. "watch" (0.20-0.40) vs. "bad"
  (>0.40).
- Vertical dashed line at the best-epoch from chart 1.

**Source:** per-epoch CSV.

**Implementation:** `plot_training_metrics.py::plot_generalization_gap`.

---

## Chart 3 — Train/val ratio + LR overlay

**Claim:** the ratio of `val_loss / train_loss` stabilizes — and any drop in
LR (from a ReduceLROnPlateau schedule) is followed by *both* losses moving
in lockstep, not just `train_loss` (which would mean the LR schedule is
just letting the model overfit harder).

**Pass thresholds:**

- Ratio asymptotes to a value between 1.0 and 2.5.
- After each LR drop, `train_loss` and `val_loss` both decrease (chart 1
  reads). If only `train_loss` drops post-LR-cut, the model is overfitting
  in the cooldown.

**X / Y:** epoch / ratio (left axis), epoch / lr (right axis, log scale).

**Source:** per-epoch CSV (`lr` column already present).

**Implementation:** `plot_training_metrics.py::plot_ratio_and_lr`.

---

## Chart 4 — Predicted vs. actual scatter with cone

**Claim:** P50 predictions track the y=x line; the [P10, P90] cone widens
*around* the line, not lopsided.

**Pass thresholds:**

- Slope of OLS fit on `(actual, p50)` is in `[0.85, 1.15]`.
- Intercept within ±20 bu/ac.
- Roughly 80% of points have `p10 ≤ actual ≤ p90` (this is exactly the
  empirical-coverage claim from chart 5).
- No "fan" pattern (predictions clustered around historical mean regardless
  of actual) — that pattern would mean the model defaulted to the prior.

**X / Y:** actual `nass_value` / predicted `yield_p50`. Error bars from
`yield_p10` / `yield_p90`.

**Layers:**

- Scatter colored by `state_fips`.
- y = x reference line.
- OLS fit line with slope/intercept in the corner.
- Annotation: `n`, RMSE, MAPE, coverage[P10, P90] (read from
  `reports/test_<year>_metrics.csv` ALL row).

**Source:** test-year predictions parquet + bundle's NASS labels.

**Implementation:** `plot_training_metrics.py::plot_pred_vs_actual`.

---

## Chart 5 — Reliability diagram for the predictive cone

**Claim:** the model is **calibrated** — when it says 80% confidence, it's
right about 80% of the time across thresholds. Coverage 0.97 (RUN-2's
leak signature) and coverage 0.30 (a too-tight cone) both lose on this
chart; only ~0.80 wins.

**Pass thresholds:**

- For nominal coverage `α ∈ {0.10, 0.20, ..., 0.90}`, the empirical
  fraction of test points inside the model's `[Q_{α/2}, Q_{1−α/2}]`
  interval lies on the y=x reference line within ±0.10.

**Note:** TFTModel only exposes [P10, P50, P90] in our config, so chart 5 in
its full form (multiple α levels) needs the predict step to draw extra
quantiles. Easiest path: re-call `predict_tft` once with
`num_samples=1000`, then compute empirical quantiles per (geoid, year)
from the raw sample array. The plotting script does this if we pass it
the *samples* parquet rather than the per-quantile parquet.

A 1-point version of chart 5 (just the [P10, P90] coverage on test 2024)
*is* available from `reports/test_<year>_metrics.csv` and is plotted by
default — that single dot lands on the y=x line at α=0.80 if calibrated.

**X / Y:** nominal interval width / empirical interval width.

**Layers:**

- y=x reference.
- Per-α point (or single dot at 0.80).
- ±0.10 shaded band around y=x.

**Source:** predictions parquet (single-α version) or raw samples (full
version).

**Implementation:** `plot_training_metrics.py::plot_reliability`.

---

## Chart 6 — Per-county residual choropleth

**Claim:** errors are **spatially unbiased** — no cluster of counties is
systematically over- or under-predicted, which is what county-specific
overfitting would look like.

**Pass thresholds:**

- No spatial autocorrelation in residuals (visually: no big red blob and
  no big blue blob; quantitatively: Moran's I < 0.2 over the test-year
  county set, optional).
- Residuals are roughly symmetric around zero (skewness in
  `[-0.5, +0.5]`).

**Geometry:** county polygons from `engine.counties.load_counties()`,
joined to test-year predictions on `geoid`. Color = `p50 − nass_value`,
diverging colormap centered at zero.

**Source:** test-year predictions parquet + counties geoparquet (already
cached by the engine).

**Implementation:** `plot_training_metrics.py::plot_residual_map` (uses
`geopandas.GeoDataFrame.plot`; degrades gracefully to an x/y scatter
colored by residual if `geopandas` isn't installed in the demo env).

---

## Chart 7 — Per-fold loss curves stacked (k-fold cross-validation)

**Claim:** the choice of validation year was **not the source of our luck**.
If you swap which year is held out, the loss curves (and best-epoch
`val_loss`) look the same.

**Pass thresholds:**

- All k curves cluster within ±0.10 of `val_loss` at the best epoch.
- All k best-epoch values land on the same epoch ±5.

**X / Y:** epoch / val_loss, one line per fold (faded for non-current).

**Source:** k per-epoch CSVs (one per fold). Requires running
`hack26-train` k times with different `--val-year` (or the proposed
`--folds` flag from the robustness doc, item #7). Even without the new
flag, this chart can be assembled from k separate runs.

**Implementation:** `plot_training_metrics.py::plot_kfold_loss_curves`
(takes a glob of CSVs).

---

## Chart 8 — Year-by-year holdout RMSE bar chart

**Claim:** the model generalizes across **time**, not just across
counties. Hold out year `Y` from training and predict it; do this for
`Y ∈ {2018, ..., 2024}` and the bar chart should be flat-ish.

**Pass thresholds:**

- All bars in 6-15 bu/ac (USDA WASDE territory).
- No bar > 2× the median (would indicate a year-specific failure).
- Coverage on each held-out year between 0.65 and 0.90.

**X / Y:** held-out year / RMSE bu/ac. Secondary panel: held-out year /
P10-P90 coverage (with horizontal reference at 0.80).

**Source:** k separate `hack26-train --test-year Y` runs. Their
`reports/test_<year>_metrics.csv` files are concatenated.

**Implementation:** `plot_training_metrics.py::plot_yearly_holdout_bars`
(takes a glob of metrics CSVs and groups by `test_year`).

---

## Order of priority if we're short on time

If we can only build three charts before the demo, build:

1. **Chart 1 — loss curves** (the obvious one; the absence of a leak is
   the first thing the judges will look for).
2. **Chart 4 — predicted vs. actual scatter with cone** (the visual
   that's most legible to a non-ML judge — "the dots are near the
   line").
3. **Chart 5 — calibration / coverage** (the only chart that *proves*
   the cone is honest, not decorative).

Charts 6 (map) and 8 (year bars) are the most rhetorically powerful but
require either the geopackage (chart 6) or k separate trainings
(chart 8) — schedule them after RUN-4 finishes.

---

## How to use the script

```bash
# All charts from a single completed run
python research/plot_training_metrics.py \
  --run-dir ~/hack26/data/derived \
  --train-csv-glob "logs/train_aug1_*.csv" \
  --predictions-parquet "reports/test_2024_predictions_aug1.parquet" \
  --metrics-csv "reports/test_2024_metrics.csv" \
  --out-dir ~/hack26/data/derived/research_plots_run3

# Loss curves only (fastest sanity check while a run is still going)
python research/plot_training_metrics.py \
  --train-csv-glob "logs/train_aug1_*.csv" \
  --out-dir /tmp/loss_only \
  --only loss_curves,gap,ratio_lr

# k-fold variant (point at multiple per-epoch CSVs)
python research/plot_training_metrics.py \
  --train-csv-glob "logs/train_aug1_run4_fold*.csv" \
  --out-dir ~/hack26/data/derived/research_plots_kfold \
  --only kfold

# Year-by-year holdout (point at multiple metrics CSVs)
python research/plot_training_metrics.py \
  --metrics-csv-glob "reports/test_*_metrics.csv" \
  --out-dir ~/hack26/data/derived/research_plots_yearly \
  --only yearly_holdout
```

The script is intentionally tolerant — pass it whatever you have, and it
skips charts whose inputs are missing, printing a one-liner per skip.
