# RUN-3 → RUN-4: making the TFT robust enough to defend at the demo

**Branch:** `research/robustness-and-metrics`  
**Date:** 2026-04-25  
**Inputs read:** `progress/mini-iowa.md` (RUN-1, RUN-2, RUN-3 plan + leak fix
notes), live console transcript from RUN-3, `software/engine/model.py`,
`software/engine/dataset.py`, `software/engine/counties.py`.

This is a written-out reasoning pass — not a refactor. Anything we accept here
gets re-PR'd as a small, reviewable change on top of `main`.

---

## 1. What RUN-3 actually told us

The pasted transcript shows the tail of the Iowa 2008-2024 `aug1` re-run
after the encoder-prior leak fix:

```
Epoch 28  train_loss=0.280  val_loss=0.532
Epoch 29  train_loss=0.295  val_loss=0.500   (logged val_loss=0.4998)
Epoch 30  train_loss=0.263  val_loss=0.527   (logged val_loss=0.5271)
Monitored metric val_loss did not improve in the last 20 records.
Best score: 0.402.  Signaling Trainer to stop.

fit complete:  forecast_date=aug1  train_series=1439  val_series=87
               params=398,707  elapsed=249.4s (4.2 min)
```

Cross-checking against the RUN-3 prediction in `progress/mini-iowa.md`:

| RUN-3 prediction | Observed | Verdict |
| ---------------- | -------- | ------- |
| `train_loss < val_loss` past epoch ~10 | yes (~0.26 vs ~0.50) | **Pass.** Leak truly gone. |
| `val_loss` plateau in 0.2–0.4 range    | best 0.402 — top of band | **Pass, but on the edge.** |
| EarlyStopping fires before epoch 120   | fired at epoch 30 (best ~10) | **Pass, but plateau is very early.** |
| `test_2024 RMSE` 6–15 bu/ac, coverage 0.65–0.85 | (not yet pasted from `reports/test_2024_metrics.csv`) | **Need the report file before we can grade this.** |

What we can read off without the metrics CSV:

1. **The fix landed.** No more 0.011 collapse, no more `val < train`. We are
   now training a real function approximator.
2. **The model is learning the ceiling of what Iowa-only gives it in 30
   epochs.** Best `val_loss` happened around epoch 10 and never improved over
   the next 20 — that's exactly what triggered the early stop. With 1,439
   training series and ~400k params, the model has 277 params per series.
   That ratio explains the gap.
3. **Generalization gap is real but moderate.** `val - train ≈ 0.27`,
   `val / train ≈ 2.0`. In z-scored quantile loss terms 0.27 is roughly the
   pinball loss of one extra ~0.5σ of error — not catastrophic, but visible.
4. **EarlyStopping with patience 20 + best at epoch 10 + cap at epoch 30**
   means we never even reached 30 effective epochs of "useful" learning.
   The right read is *the model is data-bound, not epoch-bound.*

We **don't** have evidence yet for any of:

- LR-schedule pathology (would show as a sawtooth in the per-epoch CSV).
- Static-covariate dominance (would need the variable-selection weights).
- Geographic / temporal bias (need predictions parquet + county map).
- Quantile mis-calibration (need test 2024 coverage at multiple α levels).

The plotting script in this folder generates exactly those diagnostics — the
first thing to do after RUN-4 is point it at the run dir.

---

## 2. Why the gap exists (mechanistic guesses, ranked)

In rough order of "how much of the 0.27 gap this can plausibly explain":

1. **Sample count.** 1,439 Iowa series is *one state*. The other four target
   states (Colorado, Missouri, Nebraska, Wisconsin) double-to-triple the
   distinct climates and corn productivity regimes the model sees. With
   ~5,000-7,000 training series, the same TFT capacity stops being
   over-parameterized.
2. **No weight decay.** `build_tft` passes only `{"lr": ...}` into
   `optimizer_kwargs`. Darts' default Adam runs with no L2 — every parameter
   is free to absorb idiosyncratic county/year noise. (This is the cheapest
   fix in the doc; it should land regardless of what else we do.)
3. **Encoder-prior is a single scalar per series.** The historical mean
   broadcast across 122 days is a constant the model can memorize per-county
   even with `jitter_std=5.0`. After ~10 epochs the model has learned
   `output ≈ prior + small weather correction` — and that's roughly where
   `val_loss` stops moving. A *time-varying* prior (running average of recent
   weather analog yields, or a per-county detrended baseline) would be a
   richer signal that's harder to memorize.
4. **No LR schedule.** Constant `lr=1e-3` for the whole run means once the
   loss flattens we just oscillate around the plateau. Every modern TFT
   recipe uses either ReduceLROnPlateau or a cosine schedule. With patience
   20 and a constant LR we waste 20 epochs after the plateau just to confirm
   it.
5. **Static covariates may be dominating.** The bundle has 13 static cols
   including 5 one-hot state flags (only one of which is nonzero in an
   Iowa-only run, so 4 of those columns are constant zero across the entire
   bundle). The TFT variable-selection net has to learn to ignore four
   never-varying inputs — wasteful, but mostly harmless. Worth checking the
   variable-selection weights in RUN-4 to confirm.
6. **Dropout = 0.1 is the Darts default; on small data 0.2 is normal.**
   This is a tiny lever but free.

---

## 3. Ranked changes for RUN-4

Each row has:

- **Lever** — what to change
- **Code** — the surface in `engine.model` (or `engine.dataset` / CLI flag)
- **Cost** — how risky / how much code
- **Expected effect** on `val_loss` / `coverage` / RMSE
- **Falsification** — what we'd plot to confirm or kill the change

| # | Lever | Code | Cost | Expected effect | Falsification |
| - | ----- | ---- | ---- | --------------- | ------------- |
| 1 | **Train on all 5 target states (`--states` omitted, default).** Re-build the bundle for 5 states × 2008-2024. | CLI only (`hack26-dataset`, `hack26-train` already accept this). | Low; data pull is ~1-2 h cold. | Largest single win. Closes most of the train/val gap. RMSE in the deliverable's USDA-WASDE band becomes plausible (6-10 bu/ac). | If `val_loss` plateau stays at 0.4 with 5× more series, the bottleneck isn't sample count — it's the encoder prior or the static covariates. Move to #3 / #5. |
| 2 | **Add weight_decay=1e-4 to the Adam optimizer.** | `build_tft` → `optimizer_kwargs={"lr": ..., "weight_decay": 1e-4}`. | ~3 lines. | Closes a few % of the gap on its own; multiplies the effect of #1. | If `train_loss` stops decreasing while `val_loss` doesn't move, weight decay is too high — drop to 1e-5. |
| 3 | **Bump dropout 0.1 → 0.2** (TFT-internal, between LSTM + attention). | `build_tft(dropout=0.2)` default. | 1 line. | Small but compounding regularization; flattens the per-epoch `train_loss` curve, leaves `val_loss` ~unchanged. | If `train_loss` jumps significantly (>0.1) without `val_loss` improving, drop back to 0.15. |
| 4 | **Add ReduceLROnPlateau callback** alongside EarlyStopping. Halve LR after `patience=5` epochs of no `val_loss` improvement; floor at 1e-5. | `_make_lr_schedule()` factory next to `_make_early_stopping`, passed through `pl_trainer_kwargs.callbacks`. | ~15 lines. | Adds a second "phase" of slow refinement after the initial plateau; usually yields another 0.02-0.05 on `val_loss`. | If LR drops to 1e-5 and `val_loss` is unchanged 10 epochs later, the model is already at its data-bound floor — useful negative evidence. |
| 5 | **Bump encoder-prior jitter 5.0 → 8.0 bu/ac**, OR replace constant prior with a *recent-analog mean* (mean yield over the 5 most-similar prior years for that county by GDD/precip) so the encoder window carries actual signal. | `--encoder-prior-jitter 8.0` is a one-flag change. The analog version is a new helper in `engine.model` next to `_inject_encoder_prior`. | Jitter: 0 lines. Analog prior: ~80 lines. | Forces the model to look past the prior, leans on the past covariates. Should reduce per-county overfit. | If RUN-4 with 8.0 jitter is *worse* than 5.0, the model needs the prior more than we thought; revert and consider a learned prior embedding instead. |
| 6 | **Past-covariate input dropout** — randomly mask 10% of past-covariate channels per training batch. Common time-series regularizer (StochasticDepth-style). | New `pl.Callback` that hooks `on_train_batch_start` and zeroes a random 10% of the past-covariate tensor. ~30 lines. | Acts like a feature-level dropout on the encoder. Particularly useful if the model is leaning hard on `GDD_cumulative` or `historical_mean` (we'd see this in variable-selection weights). | If variable-selection weights stay concentrated on one feature, this didn't work. |
| 7 | **Year-stratified validation.** Replace single-year validation (2023) with 3-fold rolling validation: `{train: 2008-2020, val: 2021}`, `{2008-2021, val: 2022}`, `{2008-2022, val: 2023}`. Average best epoch + best `val_loss` across folds before choosing the final retrain epoch budget. | New `--folds N` mode in `hack26-train` that loops `_train_one_pass`. | More compute (3x) but each fold is small. | Gives an honest variance estimate on `val_loss` and prevents overfitting hyperparams to 2023 specifically. | If fold-to-fold `val_loss` varies by >0.1, the model is *not* a stable estimator and we shouldn't ship it as-is. |
| 8 | **3-seed ensemble at predict time.** Train the same recipe with seeds {42, 7, 2026}, average the per-quantile predictions (or take per-quantile median across seeds). | New flag `--ensemble-seeds 42,7,2026`. Predict step averages the per-(geoid,year) quantile arrays. ~50 lines. | Almost always reduces RMSE by 5-15% and tightens P10/P90 calibration. The most reliable "free" win in time-series forecasting. | If ensemble RMSE is within noise of the single-model RMSE, the model is already at the irreducible noise floor — keep it for cone width but don't claim a win. |
| 9 | **Detrend the target** by subtracting `historical_mean_yield_bu_acre` and learning the residual. The cone gets re-added at predict time. | New helper next to `_inject_encoder_prior`; affects scaler fit, predict inverse, and evaluate paths. ~80 lines. | Reduces target variance ~3×, which usually means the same model needs ~half the params or doubles the per-epoch effective sample count. | If RMSE on detrended residuals is the same magnitude as RMSE on raw labels, the residual is the only signal there is — interesting but doesn't help. |
| 10 | **Smaller TFT (hidden_size 64 → 32).** Last resort if the gap survives #1-#9. | `--hidden-size 32` flag already exists. | Halves params to ~100k. | Removes capacity-side overfit. Trade is slower convergence on the multi-state run. | If RMSE / `val_loss` get worse, the model needed the capacity for the multi-state regime — revert. |

### Recommended RUN-4 recipe (the actual change set to ship)

If the goal is "best chance of moving from `val_loss=0.4 ± wide gap` to
`val_loss=0.3 with a tight gap`" with the smallest blast radius:

1. Items **#1 (5 states), #2 (weight_decay), #3 (dropout 0.2), #4 (LR
   schedule), #5 (jitter 8.0)** together. None of them touch the model
   architecture; #1 is data; #2/#3/#5 are flag/CLI changes; #4 is one new
   callback.
2. Skip #6/#9/#10 unless RUN-4's gap is still > 0.2.
3. Add #8 (3-seed ensemble) only after the per-seed numbers are good — it's
   noise-reducing, not bias-fixing.

Concrete CLI for RUN-4 once the changes land:

```bash
hack26-dataset \
  --start 2008 --end 2024 \
  --save-last-bundle \
  --max-fetch-workers 6 \
  --allow-download \
  -v --log-file ~/hack26/data/derived/logs/dataset_5state_2008_2024.log

hack26-train \
  --forecast-date aug1 \
  --train-years 2008-2022 \
  --val-year 2023 \
  --test-year 2024 \
  --epochs 200 \
  --patience 25 \
  --batch-size 256 \
  --num-samples 500 \
  --encoder-prior-jitter 8.0 \
  --dropout 0.2 \
  --weight-decay 1e-4 \                # new flag
  --lr-schedule plateau \              # new flag (default 'none')
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --out-dir 5state_2008_2024_aug1_run4 \
  -v --log-file ~/hack26/data/derived/logs/train_5state_2008_2024_aug1_run4.log
```

`bf16-mixed` is safe on the SageMaker T4 / A10 / L4 GPU images; if it OOMs
or gives NaNs, fall back to `32-true`.

### Minimal engine surface to add for #2 + #4 + (#6 optional)

In `software/engine/model.py`:

```python
# build_tft signature additions
def build_tft(..., weight_decay: float = 0.0, lr_schedule: str = "none", ...):
    ...
    optimizer_kwargs = {"lr": float(learning_rate)}
    if weight_decay > 0:
        optimizer_kwargs["weight_decay"] = float(weight_decay)
    ...
    pl_trainer_kwargs["callbacks"] = [...]  # already wired

# _make_lr_schedule factory next to _make_early_stopping
def _make_lr_schedule(kind: str = "plateau", patience: int = 5, factor: float = 0.5):
    if kind == "none":
        return None
    if kind == "plateau":
        from pytorch_lightning.callbacks import LearningRateMonitor
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        # Darts exposes lr_scheduler_cls / lr_scheduler_kwargs on TFTModel:
        return {
            "lr_scheduler_cls": ReduceLROnPlateau,
            "lr_scheduler_kwargs": {
                "mode": "min", "factor": factor, "patience": patience,
                "min_lr": 1e-5, "monitor": "val_loss",
            },
        }
    raise ValueError(f"unknown lr_schedule={kind!r}")
```

(Darts wires `lr_scheduler_cls` / `lr_scheduler_kwargs` into PL trainer
construction itself — no need to fight the callback list. The `monitor` key
isn't standard for `ReduceLROnPlateau`; pass `monitor="val_loss"` via
`pl_trainer_kwargs={"check_val_every_n_epoch": 1, ...}` and Darts handles
it. Confirm against the Darts version pinned in `pyproject.toml` before
shipping — call out in the PR.)

In the CLI:

```python
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--lr-schedule", default="none", choices=["none", "plateau"])
```

Total surface: ~25 lines + 2 CLI args. Well under a single PR's worth of
review effort.

---

## 4. What we will *not* change in RUN-4

Equally important — some plausible-looking knobs are traps:

- **Bigger model (hidden_size > 64).** With our sample count, more capacity
  makes the gap worse. Park this until we have ≥ 20k training series.
- **Longer encoder window.** `aug1` already encodes Apr 1 → Jul 31; nothing
  before April carries season-relevant signal in the corn belt. Lengthening
  it just adds memorization surface.
- **Different likelihood (Gaussian, Laplace).** Quantile regression on
  `[0.1, 0.5, 0.9]` is what gives us the cone the deck needs. Switching
  loses the directly-interpretable interval.
- **Removing `historical_mean_yield_bu_acre` from statics.** It's leak-free
  (computed from `< target_year` only) and is the strongest single county-
  level prior. Removing it would *increase* RMSE.
- **Augmenting on `--include-sentinel`.** NDVI/NDWI is high-value but
  Sentinel-2 pulls are slow and we don't yet have the cache pre-warmed for
  5 states × 17 years. Add it in RUN-5, not RUN-4.

---

## 5. Open questions that block the next step

1. **What's in `reports/test_2024_metrics.csv` for RUN-3?** That's the only
   thing that tells us whether the leak fix produced an honest-skill model
   or a slightly-better-than-historical-mean one. If it's
   `RMSE > 25 bu/ac`, item #5 (encoder prior) jumps to the top of the
   priority list ahead of #1 (more data). Without it, the priority order
   above is calibrated on the loss-curve evidence alone.
2. **What does the per-epoch CSV look like over all 30 epochs (not just the
   tail)?** If `val_loss` was already at 0.40 by epoch 8 and we wasted 22
   epochs at patience 20, we should drop patience to 10 in RUN-4 and ship
   the LR schedule simultaneously.
3. **Does `bf16-mixed` precision NaN on this model?** Easy to test on a
   throwaway 5-epoch run before committing to RUN-4. The doc above assumes
   yes; if it doesn't, fall back to `32-true` and the recipe still works.

The plotting script in this branch generates a `loss_curve.png` and prints
the per-epoch table, which answers (1) and (2) directly the moment the user
points it at the run directory.
