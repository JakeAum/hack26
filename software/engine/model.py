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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

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
    """Import :class:`darts.models.forecasting.tft_model.TFTModel` directly.

    Importing the public ``darts.models`` package eagerly loads every model
    in Darts' registry, including ``catboost_model``, which transitively
    imports the ``catboost`` C extension. On environments where ``catboost``
    was built against a different numpy ABI (a common mismatch on the AWS
    SageMaker conda image), that import raises::

        ValueError: numpy.dtype size changed, may indicate binary
        incompatibility. Expected 96 from C header, got 88 from PyObject

    even though we never use CatBoost. Importing the leaf submodule skips
    the registry sweep and avoids the issue.
    """
    from darts.models.forecasting.tft_model import TFTModel
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
    model.save(str(path))
    meta = {
        "forecast_date": getattr(model, "_hack26_forecast_date", None),
        "input_chunk_length": int(model.input_chunk_length),
        "output_chunk_length": int(model.output_chunk_length),
        "quantiles": list(QUANTILES),
        "train_years": getattr(model, "_hack26_train_years", None),
        "val_year": getattr(model, "_hack26_val_year", None),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("saved model -> %s   meta -> %s", path, sidecar)
    return path


def load_tft(path: Path):
    """Load a previously-saved TFTModel + sidecar metadata."""
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
    return model


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

    fit_kwargs: dict = {
        "series": train_bundle.target_series,
        "past_covariates": train_bundle.past_covariates,
        "future_covariates": train_bundle.future_covariates,
        "verbose": True,
    }
    if val_bundle is not None and val_bundle.n_series > 0:
        fit_kwargs["val_series"] = val_bundle.target_series
        fit_kwargs["val_past_covariates"] = val_bundle.past_covariates
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

    _, output_chunk = _resolve_chunk_lengths(forecast_date)

    banner(
        f"PREDICT TFT {forecast_date.upper()}  n_series={bundle.n_series}  "
        f"num_samples={num_samples}",
        logger=logger,
    )

    t0 = time.monotonic()
    predict_kwargs: dict = {
        "n": output_chunk,
        "series": bundle.target_series,
        "past_covariates": bundle.past_covariates,
        "future_covariates": bundle.future_covariates,
        "num_samples": num_samples,
        "verbose": False,
    }
    if batch_size is not None:
        predict_kwargs["batch_size"] = batch_size
    preds = model.predict(**predict_kwargs)
    if not isinstance(preds, list):
        preds = [preds]
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
    from .dataset import _logs_dir as _ld  # type: ignore
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
