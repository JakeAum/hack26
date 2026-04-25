# RUN-1: Mini Iowa training note — 2026-04-25

End-to-end TFT pipeline ran cleanly — fit → save → predict → evaluate, no
exceptions. **But the metrics are diagnostic-bad and it's worth understanding
why before scaling up.**

## Run config

- States: Iowa (single state, 99 counties)
- Bundle: 2018–2024 (`n_series=617`, `past_cols=27`, `static_cols=13`)
- Split: train 2018–2022 (457) / val 2023 (87) / test 2024 (73)
- Variant: `aug1` (input_chunk=122, output_chunk=122)
- 30 epochs, batch=64, lr=1e-3, hidden=64, heads=4, CPU
- Wall time: ~72 s fit + 12 s predict
- Artifact: `derived/models/test_iowa_mini/tft_aug1.pt` (+ sidecar)

## The numbers

```
RMSE = 184.76 bu/acre   MAPE = 86.99%   P10-P90 coverage = 0.00
```

Iowa 2024 yields sit in roughly **190–230 bu/acre**. An RMSE of ~185 means the
model is predicting values close to **zero** (residual ≈ the labels themselves).
Coverage of **0.00** confirms the cone isn't even bracketing reality — predicted
P90 is below NASS P10, on every county. This is the model collapsing to a
near-constant low value, not a noisy-but-centered estimator.

## Why it looks like this

Almost certainly the **feature scale problem**, not the architecture or the
recent fixes (clean save, float32 casts, predict-time truncation). We feed the
TFT raw, unscaled values whose magnitudes span ~6 orders:


| Feature                                  | Typical range     |
| ---------------------------------------- | ----------------- |
| `T2M`                                    | ~270–310 (Kelvin) |
| `GDD_cumulative`                         | 0 → ~3,500        |
| `PRECTOTCORR_30d_avg`                    | 0 → ~10           |
| `log_corn_area_m2` (static)              | ~18 – 25          |
| `corn_pct_of_county` (static)            | 0 – 100           |
| `historical_mean_yield_bu_acre` (static) | ~150 – 200        |
| target `yield_bu_acre`                   | ~130 – 240        |


There is **no `Scaler` wrapped around past covariates, statics, or target** —
Darts does not normalize for you. The variable-selection network's pre-scaler
`Linear` ends up with weights tuned to whatever magnitude dominates
(`GDD_cumulative` in the thousands), and the output drifts toward whatever
value that path lands on after default `nn.init` (close to zero). Quantile
regression then can't find ~195 without internally rescaling, which would take
many more epochs than 30 on 457 series.

## Supporting evidence inside this run

- `val_loss` only dropped **297 → 256 in 30 epochs** with no plateau and a
per-epoch velocity that was still **growing** at the end. That's "model
learning to cancel a scale offset," not "model learning a function."
- `train_loss` was stuck in **248 – 295** the whole time, never breaking
below 240. Healthy training would show train falling well below val.
- Predict + eval finished in 12 s — the prediction code path itself is fine;
the model just hasn't learned the right output range.

## What is **not** the problem

- **The pipeline.** Dataset → bundle → train → save → predict → evaluate now
works cleanly end-to-end. The `predict_tft` truncation fix landed
correctly; predictions came out for all 73 Iowa-2024 counties.
- **The forecast-date geometry.** `aug1` decoder lined up on Aug 1 → Nov 30
of 2024 as designed.
- **Leakage.** `historical_mean_yield_bu_acre` is computed strictly from
prior years (a leak would show suspiciously *low* val_loss, not RMSE 185).

## What to do (in priority order)

1. **DONE — `_BundleScaler` wired into `engine.model`.**
   `train_tft` now fits a sklearn `StandardScaler` (via Darts' `Scaler` with
   `global_fit=True`) on the **training** target + past covariates, and
   z-scores the per-series `static_covariates` frame using train-only
   mean/std. Future covariates stay raw (already in well-behaved ranges).
   The fitted scaler is pickled next to the checkpoint as
   `tft_<variant>.pt.scaler.pkl`; `save_tft` writes it, `load_tft` restores
   it onto `model._hack26_bundle_scaler`, and `predict_tft` inverse-transforms
   the predicted quantiles before writing the parquet. Legacy unscaled
   checkpoints still load with a warning.
2. **Bump `--epochs 120 --patience 20`** — at 30 epochs the curve is nowhere
   near convergence even before scaling.
3. **Train on more data: all 5 states, 2008–2022.** With 5,000+ series
   instead of 457 the optimizer behaves very differently.
4. **Move to GPU.** CPU is fine for the smoke; the deliverable run will be
   painfully slow without it.

Steps 1 + 2 alone, still on this Iowa-mini bundle, should drop test-2024 RMSE
into the tens of bu/acre and push P10–P90 coverage above 0.5. If that doesn't
happen after the scaler is in, the next thing to look at is the static
covariates path — but it would be surprising to need to.

## RUN-2: Next experiment — Iowa 2008-2024 on GPU

**Goal:** retrain the same `aug1` head with the full 17-year history, more
epochs, and higher patience on a GPU box, so train_loss is no longer
limited by 457 series and 30 epochs of CPU. This is still a measurement run
(test_year = 2024, val_year = 2023); the deliverable retrain comes after.

### 0. One-time on the GPU instance

```bash
cd ~/hack26
git pull
pip install -e '.[forecast]'        # darts + torch + pytorch-lightning + catboost wheel
nvidia-smi                          # confirm a GPU is visible
echo $NASS_API_KEY | head -c 4      # confirm the key is exported
```

If `nvidia-smi` is empty, stop — the rest of this section assumes a CUDA
device. (CPU works but Pass 1 will take hours instead of minutes.)

### 1. Build the 2008-2024 Iowa bundle (cold pull, ~8-15 min cold)

```bash
hack26-dataset \
  --states Iowa \
  --start 2008 --end 2024 \
  --save-last-bundle \
  --max-fetch-workers 4 \
  --allow-download \
  -v --log-file ~/hack26/data/derived/logs/dataset_iowa_2008_2024.log
```

Expect roughly **17 yr × 99 counties ≈ 1,683 series before label drops**;
final `n_series` will be lower because pre-2018 NASS county yields are
spotty for some Iowa counties. The `--save-last-bundle` flag writes
`derived/bundles/last_training_bundle.pkl` (+ meta) so the train step can
reuse it without re-fetching.

### 2. Pass 1 measurement training (GPU, 120 epochs, patience 20)

```bash
hack26-train \
  --states Iowa \
  --forecast-date aug1 \
  --train-years 2008-2022 \
  --val-year 2023 \
  --test-year 2024 \
  --epochs 120 \
  --patience 20 \
  --batch-size 128 \
  --num-samples 500 \
  --accelerator gpu \
  --devices 1 \
  --precision 32-true \
  --out-dir iowa_2008_2024_aug1 \
  -v --log-file ~/hack26/data/derived/logs/train_iowa_2008_2024_aug1.log
```

Notes / knobs to be deliberate about:

- `**--accelerator gpu --devices 1**` is the GPU switch. Drop these on CPU.
- `**--batch-size 128**` ~doubles throughput vs the smoke run; bump to 256
if the GPU has ≥ 16 GB VRAM and you don't see OOM in the per-epoch CSV.
- `**--epochs 120 --patience 20**` lets EarlyStopping pick the best epoch
instead of stopping at the artificial cap.
- `**--num-samples 500**` tightens the empirical P10/P90 from the quantile
draws at predict time (200 is fine for a smoke; the deliverable should be
≥ 500).
- Keep `--precision 32-true` for the first run; mixed precision can be
enabled later (`bf16-mixed` on Ampere+) once we trust the loss curve.
- The bundle from step 1 is reused automatically — no `--rebuild-dataset`.

### 3. What to paste back

When the train job finishes, paste / share these so we can compare:

- `~/hack26/data/derived/logs/train_iowa_2008_2024_aug1.log`
- `~/hack26/data/derived/logs/train_aug1_*.csv` (per-epoch CSV)
- `~/hack26/data/derived/reports/test_2024_metrics.csv`
- `~/hack26/data/derived/reports/test_2024_predictions_aug1.parquet`

### 4. Success criteria for this run

- `train_loss` clearly below `val_loss` by epoch ~30 (real learning, not
scale offset).
- `val_loss` plateaus before epoch 120 and EarlyStopping fires.
- `**test_2024` RMSE in the tens of bu/acre, MAPE < 15 %, P10-P90
coverage ≥ 0.5.**

If any of those miss, dig into the static-covariate path and the
hyperparameters (`hidden_size`, `dropout`, `lr`); if they hit, repeat the same recipe with
`--forecast-date all` and the four-state-plus-Iowa bundle for the
deliverable.

## RUN-2 results — leaked, not skill (Apr 25 19:03)

The Iowa 2008-2024 run completed cleanly:

- EarlyStopping at epoch 64, best `val_loss = 0.011` (z-scored quantile loss)
- 65 epochs in 8.7 min on T4 → ~8 s/epoch
- `test_2024`: **RMSE = 1.46 bu/ac, MAPE = 0.24%, P10-P90 coverage = 0.97**

Those numbers are not "we beat USDA" — they're a **target leak** signature:

- USDA WASDE state-level corn forecasts run **6-10 bu/ac RMSE**.
- A *calibrated* 80% interval should sit at ~0.80 coverage, not 0.97.
- `val_loss < train_loss` for the entire run, with both collapsing to ~0
  in 11 epochs.

### Where the leak comes from

`software/engine/dataset.py::_build_series_for_county_year` broadcasts the
**actual final yield** as a constant across every day of the season:

```python
target_values = np.full((len(season_dates), 1), float(label), dtype=np.float32)
```

Darts' TFT reads the target component as one of its encoder inputs, so the
encoder for the 2024 test bundle saw `[actual_2024_yield] × 122` for every
county. The decoder's job collapsed to "output the same constant" — an
identity map, ~zero pinball loss, and zero generalization.

**Worse — it would also break Pass 2.** The deliverable inference path
(`build_inference_dataset`) broadcasts the per-county *historical mean* as
the target (since the 2025 yield isn't known). So at inference the encoder
distribution is a different constant than what the model was trained on,
and the model — which only ever learned `output ≈ encoder_constant` — would
output something close to the historical mean and almost no skill from
weather / NDVI / GDD.

### Fix #1 + Fix #2 — implemented Apr 25 (in `engine.model`)

Both fixes are pure functions of the target series; no bundle rebuild.

- **Fix #1: `_inject_encoder_prior(target, input_chunk, jitter_std=0)`**
  overwrites the first `input_chunk` days of every target series with the
  per-county `historical_mean_yield_bu_acre` static covariate (which is
  computed strictly from `< target_year` in
  `_historical_mean_yields`, so it's leak-free). The decoder window keeps
  the actual yield as the supervised label. Called from `train_tft` AND
  `predict_tft`, so train-time and inference-time encoder distributions
  match exactly. For the deliverable inference bundle (whose targets are
  already historical means), this is a no-op.

- **Fix #2: `jitter_std=5.0` (default) at training time only.** Adds
  Gaussian noise (~17% of typical Iowa yield std) to the encoder prior so
  the model can't memorize the constant — it has to look at past
  covariates and statics to refine. Inference uses `jitter_std=0`
  (deterministic). Configurable via `--encoder-prior-jitter`.

- The `_BundleScaler` is now fit on the **prior-injected** training
  bundle, so the standardization parameters reflect the encoder
  distribution the model actually sees at predict time.

### Expected impact on RUN-3

Honest skill should land somewhere in:

- `test_2024 RMSE`: **6-15 bu/ac** (USDA WASDE territory)
- `MAPE`: **3-8%**
- `P10-P90 coverage`: **0.65-0.85** (calibrated, not over-tight)
- `val_loss` **above** `train_loss` once past epoch ~10 (normal direction)

If we instead see RMSE < 3 or coverage > 0.95 again, there's a *second*
leak we haven't found yet (most likely candidate: a static covariate
computed with `<= year` somewhere). If we see RMSE > 30 or coverage < 0.4,
the model isn't learning anything from weather/NDVI and we need to revisit
hyperparameters or the past-covariate set.

## RUN-3: re-measurement after Fix #1 + #2

Re-run the **exact same Pass-1 command** — bundle on disk is unchanged, so
no dataset rebuild. The only difference is the new injection happening
inside `train_tft` / `predict_tft`.

```bash
cd ~/hack26 && git pull
hack26-train \
  --states Iowa \
  --forecast-date aug1 \
  --train-years 2008-2022 \
  --val-year 2023 \
  --test-year 2024 \
  --epochs 120 \
  --patience 20 \
  --batch-size 128 \
  --num-samples 500 \
  --encoder-prior-jitter 5.0 \
  --accelerator gpu --devices 1 --precision 32-true \
  --out-dir iowa_2008_2024_aug1_run3 \
  -v --log-file ~/hack26/data/derived/logs/train_iowa_2008_2024_aug1_run3.log
```

Notes:

- `--out-dir iowa_2008_2024_aug1_run3` keeps RUN-2's leaky checkpoint
  intact for comparison; it lives next to the new one under
  `derived/models/`.
- The cached bundle at `derived/bundles/last_training_bundle.pkl` is
  reused (rebuild not required).
- The first epoch's log line should now read
  `encoder-prior injection: input_chunk=122  jitter_std=5.00 bu/ac
   train_series=1439  val_series=87`. If that line is missing, the new
  code didn't make it onto the box — re-pull and verify
  `git log -1 --oneline`.
- Watch for `train_loss > val_loss` after epoch ~10 (the right direction
  this time) and `val_loss` plateauing at a non-zero value (~0.2-0.4 in
  z-scored quantile loss is a healthy realistic floor; previous run hit
  0.011 because of the leak).

### What to paste back

- `~/hack26/data/derived/logs/train_iowa_2008_2024_aug1_run3.log` (tail)
- `~/hack26/data/derived/logs/train_aug1_*.csv` (per-epoch CSV)
- `~/hack26/data/derived/reports/test_2024_metrics.csv`
- `~/hack26/data/derived/reports/test_2024_predictions_aug1.parquet`

If RMSE lands in 6-15 bu/ac with coverage 0.65-0.85, kick off Pass 2
(deliverable retrain on 2008-2024 with `--forecast-date all`,
`--no-test`, `--out-dir final`).