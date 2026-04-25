"""Engine: pluggable data-source layer plus TFT model + forecast pipeline.

Every data source is a function that takes a county (geoid + geometry) and
returns a tidy DataFrame. The County Catalog (`engine.counties`) is the
single source of truth for ROIs and geometries; all other sources join on
`geoid`.

The forecast layer (``engine.dataset`` + ``engine.model`` +
``engine.analogs`` + ``engine.forecast``) consumes those tidy frames and
produces per-county / per-state corn-yield cones at the four required
forecast dates. See ``software/SPEC.md`` §12 for the full contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "load_counties",
    "load_cdl",
    "fetch_county_cdl",
    "fetch_counties_cdl",
    "fetch_county_weather",
    "fetch_counties_weather",
    "fetch_county_nass_yields",
    "fetch_counties_nass_yields",
    "fetch_nass_state_corn_forecasts",
    "nass_get",
    "build_training_dataset",
    "build_inference_dataset",
    "TrainingBundle",
    "default_last_training_bundle_path",
    "save_training_bundle",
    "load_training_bundle",
    "training_bundle_fits_train_request",
    "build_tft",
    "train_tft",
    "predict_tft",
    "evaluate_tft",
    "save_tft",
    "load_tft",
    "FORECAST_DATES",
    "FORECAST_DATE_CHUNKS",
    "analog_cone",
    "analog_cones_for_counties",
    "AnalogResult",
    "season_to_date_signature",
    "run_forecast",
    "aggregate_county_forecasts_to_state",
]

# Lazy re-exports so `python -m engine.<sub>` doesn't double-import the
# submodule (which triggers a runpy RuntimeWarning), and so importing the
# package doesn't pull rasterio / geopandas / torch into memory unnecessarily.
_LAZY: dict[str, tuple[str, str]] = {
    "load_counties":                   ("engine.counties",   "load_counties"),
    "load_cdl":                        ("engine.cdl",        "load_cdl"),
    "fetch_county_cdl":                ("engine.cdl",        "fetch_county_cdl"),
    "fetch_counties_cdl":              ("engine.cdl",        "fetch_counties_cdl"),
    "fetch_county_weather":            ("engine.weather.core", "fetch_county_weather"),
    "fetch_counties_weather":          ("engine.weather.core", "fetch_counties_weather"),
    "fetch_county_nass_yields":        ("engine.nass.core",  "fetch_county_nass_yields"),
    "fetch_counties_nass_yields":      ("engine.nass.core",  "fetch_counties_nass_yields"),
    "fetch_nass_state_corn_forecasts": ("engine.nass.core",  "fetch_nass_state_corn_forecasts"),
    "nass_get":                        ("engine.nass.core",  "nass_get"),
    # ---- forecast layer -----------------------------------------------------
    "build_training_dataset":          ("engine.dataset",    "build_training_dataset"),
    "build_inference_dataset":         ("engine.dataset",    "build_inference_dataset"),
    "TrainingBundle":                  ("engine.dataset",    "TrainingBundle"),
    "default_last_training_bundle_path": ("engine.dataset", "default_last_training_bundle_path"),
    "save_training_bundle":            ("engine.dataset",    "save_training_bundle"),
    "load_training_bundle":            ("engine.dataset",    "load_training_bundle"),
    "training_bundle_fits_train_request": ("engine.dataset", "training_bundle_fits_train_request"),
    "build_tft":                       ("engine.model",      "build_tft"),
    "train_tft":                       ("engine.model",      "train_tft"),
    "predict_tft":                     ("engine.model",      "predict_tft"),
    "evaluate_tft":                    ("engine.model",      "evaluate_tft"),
    "save_tft":                        ("engine.model",      "save_tft"),
    "load_tft":                        ("engine.model",      "load_tft"),
    "FORECAST_DATES":                  ("engine.model",      "FORECAST_DATES"),
    "FORECAST_DATE_CHUNKS":            ("engine.model",      "FORECAST_DATE_CHUNKS"),
    "analog_cone":                     ("engine.analogs",    "analog_cone"),
    "analog_cones_for_counties":       ("engine.analogs",    "analog_cones_for_counties"),
    "AnalogResult":                    ("engine.analogs",    "AnalogResult"),
    "season_to_date_signature":        ("engine.analogs",    "season_to_date_signature"),
    "run_forecast":                    ("engine.forecast",   "run_forecast"),
    "aggregate_county_forecasts_to_state": (
        "engine.forecast", "aggregate_county_forecasts_to_state",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'engine' has no attribute {name!r}")
    mod_name, attr = target
    import importlib
    return getattr(importlib.import_module(mod_name), attr)


if TYPE_CHECKING:
    from engine.analogs import (  # noqa: F401
        AnalogResult,
        analog_cone,
        analog_cones_for_counties,
        season_to_date_signature,
    )
    from engine.cdl import fetch_counties_cdl, fetch_county_cdl, load_cdl  # noqa: F401
    from engine.counties import load_counties  # noqa: F401
    from engine.dataset import (  # noqa: F401
        TrainingBundle,
        build_inference_dataset,
        build_training_dataset,
        default_last_training_bundle_path,
        load_training_bundle,
        save_training_bundle,
        training_bundle_fits_train_request,
    )
    from engine.forecast import (  # noqa: F401
        aggregate_county_forecasts_to_state,
        run_forecast,
    )
    from engine.model import (  # noqa: F401
        FORECAST_DATE_CHUNKS,
        FORECAST_DATES,
        build_tft,
        evaluate_tft,
        load_tft,
        predict_tft,
        save_tft,
        train_tft,
    )
    from engine.nass.core import (  # noqa: F401
        fetch_counties_nass_yields,
        fetch_county_nass_yields,
        fetch_nass_state_corn_forecasts,
        nass_get,
    )
    from engine.weather.core import fetch_counties_weather, fetch_county_weather  # noqa: F401
