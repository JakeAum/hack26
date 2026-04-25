"""Darts-based Temporal Fusion Transformer wrapper for corn-yield forecasting.

Four sibling models, one per forecast-date checkpoint
(``aug1`` / ``sep1`` / ``oct1`` / ``final``), each with its own encoder/decoder
window so the in-season covariate availability matches what we'll have at
inference. All four use ``QuantileRegression(quantiles=[0.1, 0.5, 0.9])`` so
the same artifact produces the **model cone** (P10 / P50 / P90) the deck needs.

Public surface:

    FORECAST_DATES               tuple of supported variant ids
    FORECAST_DATE_CHUNKS         dict variant -> (input, output) chunk lengths
    build_tft(forecast_date, ...) -> TFTModel
    train_tft(bundle, forecast_date, train_years, val_year, ...) -> TFTModel
    predict_tft(model, bundle, forecast_date) -> pd.DataFrame
    evaluate_tft(predictions, labels) -> pd.DataFrame
    save_tft(model, path)
    load_tft(path) -> TFTModel
    _BundleScaler                fits StandardScaler on train target+past+statics,
                                 saved alongside the checkpoint and applied
                                 transparently in train_tft / predict_tft.
    _main(argv)                  CLI:  hack26-train ...

All training entry points enforce the 2025-strict-holdout: any
``train_years`` / ``val_year`` / ``test_year`` referencing 2025 raises
:class:`ValueError`. The deliverable forecast (2025) goes through
``engine.forecast``, never through ``engine.model.train_tft``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from ._logging import (
    add_cli_logging_args,
    apply_cli_logging_args,
    banner,
    get_logger,
    log_environment,
)
from .dataset import (
    MAX_TRAIN_YEAR,
    MIN_TRAIN_YEAR,
    TrainingBundle,
    build_training_dataset,
    default_last_training_bundle_path,
    load_training_bundle,
    load_training_bundle_meta,
    training_bundle_fits_train_request,
)

if TYPE_CHECKING:
    from darts.models.forecasting.tft_model import TFTModel  # noqa: F401

logger = get_logger(__name__)


def _import_tft_model():
    """Return :class:`darts.models.forecasting.tft_model.TFTModel`.

    Darts' ``darts.models`` package ``__init__`` import graph pulls in
    ``catboost`` even for TFT-only use. A ``catboost`` wheel that was built
    against a different NumPy ABI (common on mixed conda + pip images) can
    raise a ``ValueError`` at import. We declare ``catboost`` in the
    ``[forecast]`` extra so resolvers can install a current wheel; if it still
    fails, ``pip install -U 'catboost>=1.2.5'`` or match NumPy to the wheel
    (e.g. ``pip install 'numpy<2'``) fixes it. Importing a leaf submodule
    does *not* skip ``darts.models``'s ``__init__`` — that is a parent package.
    """
    try:
        from darts.models.forecasting.tft_model import TFTModel
    except ValueError as e:
        msg = str(e)
        if "numpy.dtype" in msg or "incompatibility" in msg.lower():
            raise RuntimeError(
                "Darts imported CatBoost, but the CatBoost binary does not match "
                "this NumPy build (common on SageMaker/conda). Try:\n"
                "  pip install -U 'catboost>=1.2.5'\n"
                "or align NumPy to 1.26 line:\n"
                "  pip install 'numpy>=1.26,<2.0.0'\n"
                f"Original error: {e!r}"
            ) from e
        raise
    return TFTModel

# ---------------------------------------------------------------------------
# Forecast-date chunk geometry
# ---------------------------------------------------------------------------

#: Supported forecast-date variants. Order = chronological.
FORECAST_DATES: tuple[str, ...] = ("aug1", "sep1", "oct1", "final")

#: ``forecast_date -> (input_chunk_length, output_chunk_length)`` over the
#: default 244-day growing season (Apr 1 → Nov 30). Sums to 244 for the
#: in-season variants; the post-harvest ``final`` model collapses to a
#: 1-step decoder so the TFT regression is just "predict yield given full
#: season".
FORECAST_DATE_CHUNKS: dict[str, tuple[int, int]] = {
    "aug1":  (122, 122),  # Apr 1 - Jul 31  ->  Aug 1 - Nov 30
    "sep1":  (153, 91),   # Apr 1 - Aug 31  ->  Sep 1 - Nov 30
    "oct1":  (183, 61),   # Apr 1 - Sep 30  ->  Oct 1 - Nov 30
    "final": (243, 1),    # Apr 1 - Nov 29  ->  Nov 30 (point estimate)
}

#: Quantiles emitted by every model. Hard-coded so the prediction frame schema
#: doesn't drift between training runs.
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Year-range guards (mirrored from dataset)
# ---------------------------------------------------------------------------

def _validate_year_split(
    train_years: Sequence[int],
    val_year: int | None,
    test_year: int | None,
) -> None:
    """Raise if any year in the split touches 2025 or precedes available data."""
    bad: list[str] = []
    for y in train_years:
        if int(y) > MAX_TRAIN_YEAR:
            bad.append(f"train year {y} > MAX_TRAIN_YEAR={MAX_TRAIN_YEAR}")
        if int(y) < MIN_TRAIN_YEAR:
            bad.append(f"train year {y} < MIN_TRAIN_YEAR={MIN_TRAIN_YEAR}")
    if val_year is not None and int(val_year) > MAX_TRAIN_YEAR:
        bad.append(f"val_year {val_year} > MAX_TRAIN_YEAR")
    if test_year is not None and int(test_year) > MAX_TRAIN_YEAR:
        bad.append(f"test_year {test_year} > MAX_TRAIN_YEAR")
    if bad:
        raise ValueError(
            "[2025-leak-guard] year split rejected:\n  - " + "\n  - ".join(bad)
            + "\n2025 is the strict holdout for the deliverable forecast."
        )


# ---------------------------------------------------------------------------
# PyTorch Lightning callbacks
# ---------------------------------------------------------------------------

def _make_csv_epoch_logger(csv_path: Path, tag: str = ""):
    """Return a tiny PL callback that appends one CSV row per validation epoch.

    Defined as a factory so importing :mod:`engine.model` doesn't require
    ``pytorch_lightning`` to be installed (it's only a hard dep when
    actually training).
    """
    import pytorch_lightning as pl

    class CsvEpochLogger(pl.Callback):
        """Per-epoch metrics CSV — easy to paste, easy to ``pd.read_csv``."""

        def __init__(self, path: Path, run_tag: str):
            super().__init__()
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._tag = run_tag
            self._t0: float | None = None
            if not self.path.exists():
                with open(self.path, "w", encoding="utf-8") as fh:
                    fh.write(
                        "ts,epoch,train_loss,val_loss,lr,elapsed_s,"
                        "vram_alloc_gb,vram_peak_gb,tag\n"
                    )

        def on_train_epoch_start(self, trainer, pl_module):
            self._t0 = time.monotonic()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass

        def on_validation_epoch_end(self, trainer, pl_module):
            try:
                metrics = trainer.callback_metrics
                epoch = int(trainer.current_epoch)
                train_loss = float(metrics.get("train_loss", float("nan")))
                val_loss = float(metrics.get("val_loss", float("nan")))
                if trainer.optimizers:
                    lr = float(trainer.optimizers[0].param_groups[0]["lr"])
                else:
                    lr = float("nan")
                elapsed = (
                    time.monotonic() - self._t0
                    if self._t0 is not None else 0.0
                )
                vram_alloc = vram_peak = 0.0
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_alloc = round(torch.cuda.memory_allocated() / 1e9, 2)
                        vram_peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                except Exception:  # noqa: BLE001
                    pass
                ts = datetime.now().isoformat(timespec="seconds")
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(
                        f"{ts},{epoch},{train_loss:.6f},{val_loss:.6f},"
                        f"{lr:.3e},{elapsed:.2f},{vram_alloc},{vram_peak},"
                        f"{self._tag}\n"
                    )
                logger.info(
                    "epoch=%d train_loss=%.4f val_loss=%.4f lr=%.2e "
                    "elapsed=%.1fs vram_alloc=%.2fGB vram_peak=%.2fGB tag=%s",
                    epoch, train_loss, val_loss, lr, elapsed,
                    vram_alloc, vram_peak, self._tag,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("CsvEpochLogger failed for epoch: %s", exc)

    return CsvEpochLogger(csv_path, tag)


def _make_progress_callback():
    """Rich progress bar if available, otherwise PL's stock TQDM bar."""
    try:
        from pytorch_lightning.callbacks import RichProgressBar
        return RichProgressBar(leave=True)
    except Exception:  # noqa: BLE001
        try:
            from pytorch_lightning.callbacks import TQDMProgressBar
            return TQDMProgressBar(refresh_rate=10)
        except Exception:  # noqa: BLE001
            return None


def _make_early_stopping(patience: int = 8):
    from pytorch_lightning.callbacks import EarlyStopping
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _resolve_chunk_lengths(forecast_date: str) -> tuple[int, int]:
    if forecast_date not in FORECAST_DATE_CHUNKS:
        raise ValueError(
            f"unknown forecast_date {forecast_date!r}; "
            f"expected one of {list(FORECAST_DATE_CHUNKS)}"
        )
    return FORECAST_DATE_CHUNKS[forecast_date]


def build_tft(
    forecast_date: str,
    *,
    hidden_size: int = 64,
    lstm_layers: int = 1,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 16,
    batch_size: int = 64,
    n_epochs: int = 30,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    pl_callbacks: list | None = None,
    accelerator: str | None = None,
    devices: int | str = "auto",
    precision: str | int = "32-true",
):
    """Construct a fresh ``TFTModel`` for the given forecast date.

    Args:
        forecast_date: one of ``FORECAST_DATES``.
        hidden_size, lstm_layers, num_attention_heads, dropout,
        hidden_continuous_size, batch_size, n_epochs, learning_rate,
        random_state: standard TFT/PL hyperparameters.
        pl_callbacks: extra PyTorch Lightning callbacks (CSV epoch logger,
            early stopping, progress bar) to attach.
        accelerator: ``"gpu"`` / ``"cpu"`` / ``None`` (auto). PL handles the
            None case by picking the best available device.
        devices: number of devices (or ``"auto"``).
        precision: PL precision string (``"32-true"`` / ``"16-mixed"`` / ...).

    Returns:
        An unfitted ``TFTModel``.
    """
    TFTModel = _import_tft_model()
    from darts.utils.likelihood_models import QuantileRegression

    input_chunk, output_chunk = _resolve_chunk_lengths(forecast_date)

    pl_trainer_kwargs: dict = {
        "callbacks": list(pl_callbacks or []),
        "enable_model_summary": False,
        "log_every_n_steps": 25,
        "gradient_clip_val": 1.0,
    }
    if accelerator is not None:
        pl_trainer_kwargs["accelerator"] = accelerator
    if devices is not None:
        pl_trainer_kwargs["devices"] = devices
    if precision is not None:
        pl_trainer_kwargs["precision"] = precision

    model = TFTModel(
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        batch_size=batch_size,
        n_epochs=n_epochs,
        likelihood=QuantileRegression(quantiles=list(QUANTILES)),
        optimizer_kwargs={"lr": float(learning_rate)},
        pl_trainer_kwargs=pl_trainer_kwargs,
        random_state=random_state,
        use_static_covariates=True,
        add_relative_index=False,
        save_checkpoints=False,
        force_reset=True,
        model_name=f"tft_{forecast_date}",
    )
    logger.info(
        "built TFT(%s):  input_chunk=%d  output_chunk=%d  hidden=%d  heads=%d  "
        "epochs=%d  batch=%d  lr=%.1e  quantiles=%s",
        forecast_date, input_chunk, output_chunk, hidden_size,
        num_attention_heads, n_epochs, batch_size, learning_rate,
        list(QUANTILES),
    )
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_tft(model, path: Path) -> Path:
    """Save the fitted model to disk plus a sidecar metadata JSON.

    Sidecar carries enough info to reconstruct calling conventions on load
    (forecast date, chunk lengths, quantiles, training year span).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``clean=True`` strips the PL ``Trainer`` (and its callbacks) before
    # pickling. We attach a ``CsvEpochLogger`` defined inside a factory
    # function for soft-dep reasons, and pickle can't relocate that local
    # class on load. Cleaning sidesteps the issue and keeps artifacts small.
    try:
        model.save(str(path), clean=True)
    except TypeError:
        # Older Darts releases didn't expose ``clean=``; fall back and rely on
        # the upcoming load to ignore unpicklable callback state.
        model.save(str(path))
    scaler = getattr(model, "_hack26_bundle_scaler", None)
    scaler_path: Path | None = None
    if scaler is not None:
        scaler_path = _BundleScaler.sidecar_path_for(path)
        scaler.save(scaler_path)
    meta = {
        "forecast_date": getattr(model, "_hack26_forecast_date", None),
        "input_chunk_length": int(model.input_chunk_length),
        "output_chunk_length": int(model.output_chunk_length),
        "quantiles": list(QUANTILES),
        "train_years": getattr(model, "_hack26_train_years", None),
        "val_year": getattr(model, "_hack26_val_year", None),
        "has_bundle_scaler": scaler is not None,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    if scaler_path is not None:
        logger.info(
            "saved model -> %s   meta -> %s   scaler -> %s",
            path, sidecar, scaler_path,
        )
    else:
        logger.info("saved model -> %s   meta -> %s   (no scaler)", path, sidecar)
    return path


def load_tft(path: Path):
    """Load a previously-saved TFTModel + sidecar metadata + bundle scaler."""
    TFTModel = _import_tft_model()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}")
    model = TFTModel.load(str(path))
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as fh:
            meta = json.load(fh)
        for k, v in meta.items():
            setattr(model, f"_hack26_{k}", v)
        logger.info("loaded model %s  (forecast_date=%s, train_years=%s)",
                    path, meta.get("forecast_date"), meta.get("train_years"))
    else:
        logger.info("loaded model %s  (no sidecar metadata)", path)
    scaler_path = _BundleScaler.sidecar_path_for(path)
    if scaler_path.exists():
        try:
            setattr(
                model, "_hack26_bundle_scaler", _BundleScaler.load(scaler_path)
            )
            logger.info("loaded bundle scaler -> %s", scaler_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "failed to load bundle scaler %s (%s); predictions will run on "
                "raw inputs and may be wildly off-scale", scaler_path, exc,
            )
    else:
        logger.info(
            "no bundle scaler found at %s; assuming a legacy unscaled model",
            scaler_path,
        )
    return model


# ---------------------------------------------------------------------------
# Bundle scaler (fixes RMSE/coverage collapse from raw-magnitude features)
# ---------------------------------------------------------------------------

@dataclass
class _BundleScaler:
    """Per-component standardization for target, past covariates, and statics.

    The TFT is sensitive to feature magnitudes; our raw past covariates
    (``GDD_cumulative`` in the thousands, ``T2M`` in Kelvin) and statics
    (``log_corn_area_m2`` ~20, one-hot state flags ~0/1, historical mean yield
    ~190) span ~6 orders of magnitude. Without a scaler the variable-selection
    pre-scaler ``Linear`` saturates on the dominant components and the model
    collapses to near-zero output (see progress/mini-iowa.md).

    This helper:
      - fits a Darts :class:`Scaler` (sklearn ``StandardScaler``) on
        ``train_bundle.target_series`` (so the model learns z-scored yields)
      - fits a second :class:`Scaler` on ``train_bundle.past_covariates``
      - manually z-scores the per-series ``static_covariates`` DataFrame
        (Darts' ``Scaler`` deliberately ignores statics)
      - persists itself next to the model checkpoint as
        ``<ckpt>.scaler.pkl`` so :func:`load_tft` and :func:`predict_tft`
        can transparently re-apply / inverse the transform.

    Future covariates (``doy_sin/cos``, ``week_sin/cos``, ``month``,
    ``days_until_end_of_season``) are intentionally **not** scaled — they're
    already in well-behaved ranges and the calendar signal is interpretable.
    """

    target_scaler: Any = None
    past_scaler: Any = None
    static_means: pd.Series = field(default_factory=lambda: pd.Series(dtype="float32"))
    static_stds: pd.Series = field(default_factory=lambda: pd.Series(dtype="float32"))
    static_cols: list[str] = field(default_factory=list)

    @staticmethod
    def _make_value_scaler():
        from darts.dataprocessing.transformers import Scaler
        from sklearn.preprocessing import StandardScaler
        try:
            return Scaler(scaler=StandardScaler(), global_fit=True)
        except TypeError:
            # Older Darts releases didn't expose ``global_fit``; fall back to
            # the default per-series fit (still better than no scaling).
            return Scaler(scaler=StandardScaler())

    @classmethod
    def fit(cls, train_bundle: TrainingBundle) -> "_BundleScaler":
        """Fit on training data only (val/test are transformed, not fit)."""
        if train_bundle.n_series == 0:
            raise ValueError("cannot fit _BundleScaler on an empty bundle")

        tgt = cls._make_value_scaler()
        tgt.fit(train_bundle.target_series)

        past = cls._make_value_scaler()
        past.fit(train_bundle.past_covariates)

        static_frames = [
            ts.static_covariates for ts in train_bundle.target_series
            if ts.static_covariates is not None
        ]
        if static_frames:
            stack = pd.concat(static_frames, ignore_index=True)
            means = stack.mean(numeric_only=True).astype("float32")
            stds = stack.std(numeric_only=True).astype("float32")
            stds = stds.replace(0.0, 1.0).fillna(1.0)
            cols = [c for c in stack.columns if c in means.index]
        else:
            means = pd.Series(dtype="float32")
            stds = pd.Series(dtype="float32")
            cols = []

        logger.info(
            "fit _BundleScaler: target+past = StandardScaler (global_fit), "
            "static cols z-scored = %d", len(cols),
        )
        return cls(
            target_scaler=tgt,
            past_scaler=past,
            static_means=means,
            static_stds=stds,
            static_cols=cols,
        )

    def _scale_static_frame(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or not self.static_cols:
            return df
        out = df.copy()
        for c in self.static_cols:
            if c not in out.columns:
                continue
            mean = float(self.static_means.get(c, 0.0))
            std = float(self.static_stds.get(c, 1.0)) or 1.0
            out[c] = (out[c].astype("float32") - mean) / std
        return out.astype("float32")

    def transform_target_series(self, series_list: Sequence) -> list:
        """Scale target values **and** z-score the attached static frame."""
        if not series_list:
            return []
        scaled = self.target_scaler.transform(list(series_list))
        if not isinstance(scaled, list):
            scaled = [scaled]
        out: list = []
        for ts in scaled:
            sc = ts.static_covariates
            new_sc = self._scale_static_frame(sc)
            if new_sc is not None and (sc is None or not new_sc.equals(sc)):
                ts = ts.with_static_covariates(new_sc)
            out.append(ts)
        return out

    def transform_past(self, past_list: Sequence) -> list:
        if not past_list:
            return []
        scaled = self.past_scaler.transform(list(past_list))
        return scaled if isinstance(scaled, list) else [scaled]

    def inverse_transform_predictions(self, preds: Sequence) -> list:
        """Undo target scaling on a list of predicted ``TimeSeries``."""
        if not preds:
            return []
        out = self.target_scaler.inverse_transform(list(preds))
        return out if isinstance(out, list) else [out]

    @staticmethod
    def sidecar_path_for(model_path: Path) -> Path:
        model_path = Path(model_path)
        return model_path.with_suffix(model_path.suffix + ".scaler.pkl")

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_scaler": self.target_scaler,
            "past_scaler": self.past_scaler,
            "static_means": self.static_means,
            "static_stds": self.static_stds,
            "static_cols": self.static_cols,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path) -> "_BundleScaler":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(**d)


# ---------------------------------------------------------------------------
# Encoder-prior injection (fixes the broadcast-target leak; see notes below)
# ---------------------------------------------------------------------------
#
# Why this exists
# ---------------
# The bundle's target series for a labeled (county, year) is
# ``[actual_NASS_yield] x season_days`` — a constant. With Darts' TFT the
# encoder reads the target component as one of its inputs, so the model can
# trivially read the answer off the encoder window and the supervised loss
# collapses to ~zero (``test_2024 RMSE = 1.46 bu/ac, MAPE = 0.24%`` — see
# progress/mini-iowa.md). At deliverable time the encoder target is the
# per-county *historical mean* (built by ``build_inference_dataset``), so the
# trained model would also be pushed wildly out-of-distribution.
#
# Fix #1: inject the historical mean into the encoder window at train AND
#         predict time, so the encoder distribution is identical in both
#         regimes. The decoder window keeps the actual yield (the supervised
#         label) so the model still learns to forecast.
# Fix #2: at training time, perturb the prior with small Gaussian noise so
#         the model treats it as a noisy hint rather than a perfect prior
#         (a regularizer; pass ``jitter_std=0`` at inference).
#
# Both fixes are pure functions of the target series — no bundle rebuild.

_HISTORICAL_MEAN_STATIC_COL = "historical_mean_yield_bu_acre"


def _inject_encoder_prior(
    target_series: Sequence,
    input_chunk: int,
    *,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list:
    """Replace the first ``input_chunk`` steps of each target series with
    the per-series ``historical_mean_yield_bu_acre`` static covariate.

    Args:
        target_series: list of Darts ``TimeSeries`` (one per (geoid, year)).
        input_chunk: encoder length for the variant being trained / predicted
            (``aug1`` -> 122d, ``sep1`` -> 153d, ``oct1`` -> 183d,
             ``final`` -> 243d). Determines how many leading days are
            overwritten.
        jitter_std: standard deviation (bu/ac) of Gaussian noise added to the
            prior block. ``> 0`` only at training time; the helper is
            deterministic at ``0.0``.
        rng: optional ``np.random.Generator`` for reproducibility (a fresh
            ``default_rng(42)`` is used if ``None``).

    Returns:
        A new list of ``TimeSeries`` with the encoder window overwritten,
        decoder window untouched, static_covariates preserved verbatim.

    Notes:
        - Series whose static frame lacks ``historical_mean_yield_bu_acre``
          (or whose static frame is ``None``) are passed through unchanged
          and a single warning is logged at the end.
        - Series shorter than ``input_chunk`` get their entire value array
          overwritten with the prior, which is the correct behavior for the
          ``predict_tft`` path (where we already truncated to the encoder).
    """
    from darts import TimeSeries

    if not target_series:
        return []
    if rng is None:
        rng = np.random.default_rng(42)

    out: list = []
    n_skipped = 0
    for ts in target_series:
        sc = ts.static_covariates
        if sc is None or _HISTORICAL_MEAN_STATIC_COL not in sc.columns:
            n_skipped += 1
            out.append(ts)
            continue

        prior = float(sc[_HISTORICAL_MEAN_STATIC_COL].iloc[0])
        n = min(int(input_chunk), len(ts))
        if n <= 0:
            out.append(ts)
            continue

        values = ts.values(copy=True).astype(np.float32)
        prior_block = np.full((n, values.shape[1]), prior, dtype=np.float32)
        if jitter_std > 0:
            prior_block = prior_block + rng.normal(
                loc=0.0, scale=float(jitter_std), size=prior_block.shape
            ).astype(np.float32)
        values[:n, :] = prior_block

        try:
            new_ts = ts.with_values(values)
        except AttributeError:
            new_ts = TimeSeries.from_times_and_values(
                times=ts.time_index,
                values=values,
                columns=list(ts.columns),
                fill_missing_dates=False,
                freq=ts.freq_str,
                static_covariates=sc,
            )
        out.append(new_ts)

    if n_skipped:
        logger.warning(
            "_inject_encoder_prior: %d/%d series had no "
            "historical_mean_yield_bu_acre static covariate and were passed "
            "through unchanged (encoder leak still present for those series).",
            n_skipped, len(target_series),
        )
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _model_param_count(model) -> int:
    try:
        return sum(p.numel() for p in model.model.parameters())
    except Exception:  # noqa: BLE001
        try:
            return sum(p.numel() for p in model._model.parameters())
        except Exception:  # noqa: BLE001
            return -1


def train_tft(
    bundle: TrainingBundle,
    forecast_date: str,
    train_years: Sequence[int],
    val_year: int | None = None,
    *,
    n_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    early_stopping_patience: int = 8,
    epoch_csv_path: Path | None = None,
    accelerator: str | None = None,
    devices: int | str = "auto",
    precision: str | int = "32-true",
    random_state: int = 42,
    encoder_prior_jitter_std: float = 5.0,
):
    """Train one TFTModel variant on the year-split slice of ``bundle``.

    Args:
        bundle: full :class:`TrainingBundle` (training + val years).
        forecast_date: which variant (``aug1`` / ``sep1`` / ``oct1`` / ``final``).
        train_years: explicit list of training years (e.g. ``range(2008, 2023)``).
        val_year: optional year for early-stopping; ``None`` disables it.
        epoch_csv_path: where to write the per-epoch CSV. ``None`` -> default
            under ``~/hack26/data/derived/logs/``.
        accelerator/devices/precision: forwarded to PL trainer.
        encoder_prior_jitter_std: bu/ac standard deviation of Gaussian noise
            added to the encoder-window historical-mean prior at training
            time (Fix #2 in :func:`_inject_encoder_prior`). ``0.0`` disables
            jitter and uses the deterministic prior. Default ``5.0`` is
            ~17 % of the typical Iowa cross-county yield std and acts as a
            mild regularizer.

    Returns:
        Fitted ``TFTModel`` with sidecar attributes
        ``_hack26_forecast_date``, ``_hack26_train_years``, ``_hack26_val_year``.

    Raises:
        ValueError: if any year in the split is 2025 (or > MAX_TRAIN_YEAR).
    """
    train_years = sorted({int(y) for y in train_years})
    _validate_year_split(train_years, val_year, None)

    banner(
        f"TRAIN TFT {forecast_date.upper()}  "
        f"train={train_years[0]}-{train_years[-1]}  val={val_year}",
        logger=logger,
    )

    train_bundle = bundle.filter_by_year(train_years)
    val_bundle = (
        bundle.filter_by_year([val_year]) if val_year is not None else None
    )
    logger.info(
        "split: train_series=%d  val_series=%s  past_cols=%d  static_cols=%d",
        train_bundle.n_series,
        val_bundle.n_series if val_bundle is not None else "0",
        len(bundle.past_covariate_cols),
        len(bundle.static_covariate_cols),
    )
    if train_bundle.n_series == 0:
        raise RuntimeError("no training series after year filter")

    callbacks: list = []
    pb = _make_progress_callback()
    if pb is not None:
        callbacks.append(pb)
    if val_bundle is not None and val_bundle.n_series > 0 and early_stopping_patience > 0:
        callbacks.append(_make_early_stopping(patience=early_stopping_patience))
    if epoch_csv_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        from ._logging import _logs_dir
        epoch_csv_path = _logs_dir() / f"train_{forecast_date}_{ts}.csv"
    callbacks.append(_make_csv_epoch_logger(epoch_csv_path, tag=forecast_date))
    logger.info("per-epoch CSV: %s", epoch_csv_path)

    model = build_tft(
        forecast_date=forecast_date,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        random_state=random_state,
        pl_callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
    )

    # Encoder-prior injection (Fix #1 + Fix #2 from progress/mini-iowa.md).
    # Without this, the encoder reads the actual yield off the broadcast
    # target and the supervised loss collapses to ~zero (test_2024 leak).
    # We overwrite the first ``input_chunk`` days of every train + val target
    # series with the per-county historical mean (matching what
    # build_inference_dataset does for the deliverable 2025 bundle), plus
    # mild Gaussian jitter on the train side as a regularizer.
    input_chunk_for_prior, _ = _resolve_chunk_lengths(forecast_date)
    train_rng = np.random.default_rng(int(random_state))
    train_target_series = _inject_encoder_prior(
        train_bundle.target_series,
        input_chunk=input_chunk_for_prior,
        jitter_std=float(encoder_prior_jitter_std),
        rng=train_rng,
    )
    val_target_series = (
        _inject_encoder_prior(
            val_bundle.target_series,
            input_chunk=input_chunk_for_prior,
            jitter_std=0.0,
        )
        if val_bundle is not None and val_bundle.n_series > 0
        else None
    )
    logger.info(
        "encoder-prior injection: input_chunk=%d  jitter_std=%.2f bu/ac  "
        "train_series=%d  val_series=%s",
        input_chunk_for_prior, float(encoder_prior_jitter_std),
        len(train_target_series),
        len(val_target_series) if val_target_series is not None else "0",
    )

    # Build a *prior-injected* TrainingBundle view so the scaler fits on the
    # same encoder distribution the model will see at inference. We reuse the
    # original bundle's other columns and replace just the target_series list.
    train_bundle_for_scaler = TrainingBundle(
        target_series=train_target_series,
        past_covariates=train_bundle.past_covariates,
        future_covariates=train_bundle.future_covariates,
        static_covariates=train_bundle.static_covariates,
        series_index=train_bundle.series_index,
        past_covariate_cols=list(train_bundle.past_covariate_cols),
        static_covariate_cols=list(train_bundle.static_covariate_cols),
    )

    # Standardize target + past + statics on the *training* slice so the
    # model sees roughly N(0, 1) inputs instead of raw Kelvin / GDD-cumulative
    # magnitudes. Predictions are inverse-transformed back to bu/acre in
    # ``predict_tft`` via the same fitted scaler.
    bundle_scaler = _BundleScaler.fit(train_bundle_for_scaler)
    train_targets = bundle_scaler.transform_target_series(train_target_series)
    train_past = bundle_scaler.transform_past(train_bundle.past_covariates)

    fit_kwargs: dict = {
        "series": train_targets,
        "past_covariates": train_past,
        "future_covariates": train_bundle.future_covariates,
        "verbose": True,
    }
    if val_target_series is not None:
        fit_kwargs["val_series"] = bundle_scaler.transform_target_series(
            val_target_series
        )
        fit_kwargs["val_past_covariates"] = bundle_scaler.transform_past(
            val_bundle.past_covariates
        )
        fit_kwargs["val_future_covariates"] = val_bundle.future_covariates

    t0 = time.monotonic()
    model.fit(**fit_kwargs)
    elapsed = time.monotonic() - t0

    n_params = _model_param_count(model)
    logger.info(
        "fit complete:  forecast_date=%s  train_series=%d  val_series=%s  "
        "params=%s  elapsed=%.1fs (%.1f min)",
        forecast_date, train_bundle.n_series,
        val_bundle.n_series if val_bundle else 0,
        f"{n_params:,}" if n_params > 0 else "?",
        elapsed, elapsed / 60.0,
    )

    setattr(model, "_hack26_forecast_date", forecast_date)
    setattr(model, "_hack26_train_years", train_years)
    setattr(model, "_hack26_val_year", val_year)
    setattr(model, "_hack26_bundle_scaler", bundle_scaler)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_tft(
    model,
    bundle: TrainingBundle,
    forecast_date: str,
    *,
    num_samples: int = 200,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Run a fitted TFTModel over every series in ``bundle``.

    Returns a DataFrame with one row per (geoid, year):

        geoid, year, forecast_date, yield_p10, yield_p50, yield_p90,
        yield_mean, yield_std

    The yield value is taken from the LAST decoder step for each series
    (target was broadcast to a constant during training, so any decoder step
    converges to the same answer; using the last is the most stable).
    """
    if bundle.n_series == 0:
        raise ValueError("inference bundle is empty")

    input_chunk, output_chunk = _resolve_chunk_lengths(forecast_date)

    banner(
        f"PREDICT TFT {forecast_date.upper()}  n_series={bundle.n_series}  "
        f"num_samples={num_samples}",
        logger=logger,
    )

    # Darts predicts ``n`` steps **after the end** of each input series. Our
    # bundle stores full-season series (Apr 1 -> Nov 30); feeding them whole
    # would ask the decoder for Dec 1 -> Mar 31 of the *next* year and trip
    # the "future_covariates do not extend far enough" check. Truncate each
    # target to ``input_chunk_length`` so the decoder lands inside the same
    # growing season:
    #   aug1  : last input = Jul 31  -> output covers Aug 1 -> Nov 30 (122d)
    #   sep1  : last input = Aug 31  -> output covers Sep 1 -> Nov 30  (91d)
    #   oct1  : last input = Sep 30  -> output covers Oct 1 -> Nov 30  (61d)
    #   final : last input = Nov 29  -> output covers Nov 30           (1d)
    truncated_targets = []
    for ts in bundle.target_series:
        if len(ts) <= input_chunk:
            truncated_targets.append(ts)
        else:
            truncated_targets.append(ts[:input_chunk])

    # Encoder-prior injection — must match what train_tft did. For a
    # *measurement* (test_year) bundle the truncated target still contains
    # ``[actual_yield] x input_chunk`` (the leak); for the deliverable
    # inference bundle from build_inference_dataset it already contains
    # ``[historical_mean] x input_chunk``, in which case this call is a
    # no-op. ``jitter_std=0`` because we want deterministic predictions.
    truncated_targets = _inject_encoder_prior(
        truncated_targets,
        input_chunk=input_chunk,
        jitter_std=0.0,
    )

    # Apply the same StandardScaler the model was trained against (target +
    # past + statics). Future covariates remain raw — the calendar features
    # are already well-behaved.
    bundle_scaler = getattr(model, "_hack26_bundle_scaler", None)
    if bundle_scaler is None:
        logger.warning(
            "predict_tft: no _hack26_bundle_scaler attached to model; "
            "inputs will be passed through raw and predictions left in the "
            "model's training scale (likely meaningless if train_tft was run "
            "with scaling)."
        )
        scaled_targets = truncated_targets
        scaled_past = bundle.past_covariates
    else:
        scaled_targets = bundle_scaler.transform_target_series(truncated_targets)
        scaled_past = bundle_scaler.transform_past(bundle.past_covariates)

    t0 = time.monotonic()
    predict_kwargs: dict = {
        "n": output_chunk,
        "series": scaled_targets,
        "past_covariates": scaled_past,
        "future_covariates": bundle.future_covariates,
        "num_samples": num_samples,
        "verbose": False,
    }
    if batch_size is not None:
        predict_kwargs["batch_size"] = batch_size
    preds = model.predict(**predict_kwargs)
    if not isinstance(preds, list):
        preds = [preds]
    if bundle_scaler is not None:
        preds = bundle_scaler.inverse_transform_predictions(preds)
    elapsed = time.monotonic() - t0
    logger.info("predict done in %.1fs (%.1f series/s)",
                elapsed, bundle.n_series / max(elapsed, 1e-9))

    rows: list[dict] = []
    for i, ts in enumerate(preds):
        idx_row = bundle.series_index.iloc[i]
        # all_values shape: (n_timesteps, n_components, n_samples)
        values = ts.all_values(copy=False)
        last_step = values[-1, 0, :]  # last timestep, single component, all samples
        rows.append({
            "geoid": str(idx_row["geoid"]),
            "year": int(idx_row["year"]),
            "forecast_date": forecast_date,
            "yield_p10": float(np.percentile(last_step, 10)),
            "yield_p50": float(np.percentile(last_step, 50)),
            "yield_p90": float(np.percentile(last_step, 90)),
            "yield_mean": float(last_step.mean()),
            "yield_std": float(last_step.std(ddof=0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tft(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-state RMSE, MAPE, and [P10, P90] coverage.

    Args:
        predictions: output of :func:`predict_tft` (one row per (geoid, year)).
        labels: DataFrame with columns ``geoid, year, nass_value`` (the
            realized NASS final yield).

    Returns:
        DataFrame with per-(state_fips, forecast_date) RMSE, MAPE, coverage,
        and a final ``ALL`` aggregate row.
    """
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.merge(
        labels[["geoid", "year", "nass_value"]],
        on=["geoid", "year"], how="inner",
    )
    if df.empty:
        logger.warning("evaluate_tft: no overlap between predictions and labels")
        return pd.DataFrame()
    df["state_fips"] = df["geoid"].str[:2]
    df["err"] = df["yield_p50"] - df["nass_value"]
    df["abs_err"] = df["err"].abs()
    df["sq_err"] = df["err"] ** 2
    df["pct_err"] = df["abs_err"] / df["nass_value"].replace(0, np.nan)
    df["in_cone"] = (df["nass_value"] >= df["yield_p10"]) & (
        df["nass_value"] <= df["yield_p90"]
    )

    def _agg(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": int(len(group)),
            "rmse_bu_acre": float(np.sqrt(group["sq_err"].mean())),
            "mape_pct": float(100.0 * group["pct_err"].mean()),
            "p10_p90_coverage": float(group["in_cone"].mean()),
            "label_mean": float(group["nass_value"].mean()),
            "p50_mean": float(group["yield_p50"].mean()),
        })

    by_state = (
        df.groupby(["forecast_date", "state_fips"], as_index=False)
          .apply(_agg, include_groups=False)
    )
    overall = (
        df.groupby(["forecast_date"], as_index=False)
          .apply(lambda g: pd.concat([
              pd.Series({"state_fips": "ALL"}), _agg(g)
          ]), include_groups=False)
    )
    return pd.concat([by_state, overall], ignore_index=True)


# ---------------------------------------------------------------------------
# CLI: hack26-train
# ---------------------------------------------------------------------------

def _parse_year_range(s: str) -> list[int]:
    """Parse '2008-2022' or '2008,2010,2014' into a sorted list of ints."""
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return sorted({int(x) for x in s.split(",") if x.strip()})


def _resolve_models_dir(out_dir: str) -> Path:
    from ._logging import _logs_dir as _ld
    # Use the same data root convention as logs.
    root = _ld().parent  # derived/
    d = root / "models" / out_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _train_one_pass(
    bundle: TrainingBundle,
    forecast_dates: Sequence[str],
    train_years: Sequence[int],
    val_year: int | None,
    test_year: int | None,
    out_dir: Path,
    args,
) -> None:
    """Fit + (optionally) evaluate every requested forecast-date variant."""
    test_bundle = (
        bundle.filter_by_year([test_year]) if test_year is not None else None
    )
    eval_pieces: list[pd.DataFrame] = []

    for fd in forecast_dates:
        model = train_tft(
            bundle,
            forecast_date=fd,
            train_years=train_years,
            val_year=val_year,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_heads,
            dropout=args.dropout,
            early_stopping_patience=args.patience,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            random_state=args.seed,
            encoder_prior_jitter_std=getattr(args, "encoder_prior_jitter", 5.0),
        )
        ckpt = out_dir / f"tft_{fd}.pt"
        save_tft(model, ckpt)

        if test_bundle is not None and test_bundle.n_series > 0:
            preds = predict_tft(
                model, test_bundle, forecast_date=fd,
                num_samples=args.num_samples,
            )
            labels = bundle.series_index.rename(columns={"label": "nass_value"})[
                ["geoid", "year", "nass_value"]
            ]
            metrics = evaluate_tft(preds, labels)
            metrics["forecast_date"] = fd
            eval_pieces.append(metrics)
            preds_path = out_dir.parent.parent / "reports" / (
                f"test_{test_year}_predictions_{fd}.parquet"
            )
            preds_path.parent.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(preds_path, index=False)
            logger.info("wrote test-year predictions -> %s", preds_path)

    if eval_pieces:
        all_metrics = pd.concat(eval_pieces, ignore_index=True)
        report_path = out_dir.parent.parent / "reports" / (
            f"test_{test_year}_metrics.csv"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        all_metrics.to_csv(report_path, index=False)
        logger.info("wrote test-year metrics CSV -> %s", report_path)
        for _, r in all_metrics.iterrows():
            logger.info(
                "metric: forecast=%s  state=%s  n=%d  RMSE=%.2f  "
                "MAPE=%.2f%%  cov[P10,P90]=%.2f",
                r["forecast_date"], r["state_fips"], int(r["n"]),
                float(r["rmse_bu_acre"]), float(r["mape_pct"]),
                float(r["p10_p90_coverage"]),
            )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one or more TFT models for corn-yield forecasting."
    )
    parser.add_argument("--forecast-date", default="all",
                        choices=["all", *FORECAST_DATES],
                        help="Which forecast-date variant(s) to train.")
    parser.add_argument("--train-years", default=f"{MIN_TRAIN_YEAR}-2022",
                        help="Year range or comma-list, e.g. '2008-2022' or "
                             "'2008,2010,2012'.")
    parser.add_argument("--val-year", type=int, default=2023,
                        help="Year for early-stopping. Pass --no-val to "
                             "disable early stopping entirely.")
    parser.add_argument("--no-val", action="store_true",
                        help="Disable val_year / early stopping.")
    parser.add_argument("--test-year", type=int, default=None,
                        help="Optional out-of-sample year to evaluate on "
                             "after training (e.g. 2024 for the deck number).")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip test-year eval (Pass 2 deliverable mode).")
    parser.add_argument("--states", nargs="+", default=None, metavar="STATE",
                        help="Subset to specific states. Omit for all 5.")
    parser.add_argument("--out-dir", default="measurement",
                        help="Subdirectory under "
                             "~/hack26/data/derived/models/ for checkpoints.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--encoder-prior-jitter", type=float, default=5.0,
        help=(
            "bu/ac standard deviation of Gaussian noise added to the encoder-"
            "window historical-mean prior at TRAINING time (Fix #2). 0.0 "
            "disables jitter and uses the deterministic prior. Default 5.0 "
            "is ~17%% of the typical Iowa cross-county yield std."
        ),
    )
    parser.add_argument("--patience", type=int, default=8,
                        help="Early-stopping patience (epochs).")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Quantile samples to draw at test-time prediction.")
    parser.add_argument("--accelerator", default=None,
                        help="PL accelerator: 'gpu', 'cpu', or None for auto.")
    parser.add_argument("--devices", default="auto",
                        help="PL devices argument; defaults to 'auto'.")
    parser.add_argument("--precision", default="32-true",
                        help="PL precision: '32-true', '16-mixed', etc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-sentinel", action="store_true",
                        help="Pull NDVI/NDWI from Sentinel-2 (slow).")
    parser.add_argument("--no-smap", action="store_true",
                        help="Skip SMAP soil moisture columns.")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download of POWER/SMAP/CDL/NASS.")
    parser.add_argument("--allow-download", action="store_true",
                        help="Permit CDL raster downloads from USDA when a "
                             "year is missing from the data root. Without "
                             "this flag missing CDL years are skipped (the "
                             "static covariates fall back to the nearest "
                             "available year).")
    parser.add_argument(
        "--dataset-bundle",
        type=Path,
        default=None,
        metavar="PATH",
        help="Load a pickled TrainingBundle from PATH (from hack26-dataset "
             "--save-bundle). If omitted, tries the last bundle from "
             "hack26-dataset --save-last-bundle when present and compatible.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Ignore any on-disk training bundle and re-run the full "
             "weather/CDL/NASS pull (same as needing a fresh build).",
    )
    parser.add_argument(
        "--max-fetch-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel I/O for weather (per county), CDL (per year), and NASS "
             "(per state). Use 1 for the legacy fully sequential pull.",
    )
    add_cli_logging_args(parser)
    args = parser.parse_args(argv)

    log_path = apply_cli_logging_args(args, tag="train")
    log_environment(logger)
    logger.info("rotated log file: %s", log_path)
    logger.info("argv: %s", " ".join(sys.argv))

    train_years = _parse_year_range(args.train_years)
    val_year = None if args.no_val else args.val_year
    test_year = None if args.no_test else args.test_year

    forecast_dates = (
        list(FORECAST_DATES) if args.forecast_date == "all"
        else [args.forecast_date]
    )
    logger.info(
        "year split: train=%s  val=%s  test=%s  out_dir=%s  variants=%s",
        f"{train_years[0]}-{train_years[-1]}" if train_years else "<empty>",
        val_year, test_year, args.out_dir, forecast_dates,
    )

    try:
        _validate_year_split(train_years, val_year, test_year)
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    end_year = max([*train_years, val_year or 0, test_year or 0])
    logger.info("dataset will pull years %d-%d", MIN_TRAIN_YEAR, end_year)

    required_years = {int(y) for y in train_years}
    if val_year is not None:
        required_years.add(int(val_year))
    if test_year is not None:
        required_years.add(int(test_year))
    required_years = {y for y in required_years if y <= MAX_TRAIN_YEAR}

    from engine.counties import _resolve_states

    resolved_states = _resolve_states(args.states)

    bundle: TrainingBundle | None = None
    try_cache = (
        not args.refresh
        and not args.rebuild_dataset
    )
    candidate: Path | None = None
    if try_cache:
        if args.dataset_bundle is not None:
            candidate = Path(args.dataset_bundle)
        else:
            candidate = default_last_training_bundle_path()

    if try_cache and candidate is not None and candidate.is_file():
        try:
            meta = load_training_bundle_meta(candidate)
            loaded = load_training_bundle(candidate)
            ok, reason = training_bundle_fits_train_request(
                loaded,
                meta,
                states_fips=resolved_states,
                required_years=required_years,
                include_sentinel=args.include_sentinel,
                include_smap=not args.no_smap,
            )
            if ok:
                bundle = loaded
                logger.info(
                    "using cached training bundle from %s (n_series=%d)",
                    candidate, bundle.n_series,
                )
            else:
                logger.warning(
                    "cached training bundle incompatible: %s — rebuilding from sources",
                    reason,
                )
        except Exception as exc:  # noqa: BLE001 - log and rebuild
            logger.warning(
                "could not load training bundle from %s (%s) — rebuilding from sources",
                candidate, exc,
            )

    if bundle is None:
        bundle = build_training_dataset(
            states=args.states,
            start_year=MIN_TRAIN_YEAR,
            end_year=end_year,
            include_sentinel=args.include_sentinel,
            include_smap=not args.no_smap,
            refresh=args.refresh,
            allow_download=args.allow_download,
            max_fetch_workers=int(args.max_fetch_workers),
        )
    logger.info("[2025-leak-guard] year_split: train=%s val=%s test=%s; "
                "2025_in_data=%s",
                train_years, val_year, test_year,
                (bundle.series_index["year"].astype(int) == 2025).any()
                if not bundle.series_index.empty else False)

    out_dir = _resolve_models_dir(args.out_dir)
    logger.info("checkpoint directory: %s", out_dir)

    _train_one_pass(
        bundle=bundle,
        forecast_dates=forecast_dates,
        train_years=train_years,
        val_year=val_year,
        test_year=test_year,
        out_dir=out_dir,
        args=args,
    )

    logger.info("training pipeline finished. log file: %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
