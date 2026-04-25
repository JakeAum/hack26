"""Engine: pluggable data-source layer.

Every source is a function that takes a county (geoid + geometry) and returns
a tidy DataFrame. The County Catalog (`engine.counties`) is the single source
of truth for ROIs and geometries; all other sources join on `geoid`.
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
]

# Lazy re-exports so `python -m engine.<sub>` doesn't double-import the
# submodule (which triggers a runpy RuntimeWarning), and so importing the
# package doesn't pull rasterio / geopandas into memory unnecessarily.
_LAZY: dict[str, tuple[str, str]] = {
    "load_counties":           ("engine.counties",       "load_counties"),
    "load_cdl":                ("engine.cdl",            "load_cdl"),
    "fetch_county_cdl":        ("engine.cdl",            "fetch_county_cdl"),
    "fetch_counties_cdl":      ("engine.cdl",            "fetch_counties_cdl"),
    "fetch_county_weather":    ("engine.weather.core",   "fetch_county_weather"),
    "fetch_counties_weather":  ("engine.weather.core",   "fetch_counties_weather"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'engine' has no attribute {name!r}")
    mod_name, attr = target
    import importlib
    return getattr(importlib.import_module(mod_name), attr)


if TYPE_CHECKING:
    from engine.cdl import fetch_counties_cdl, fetch_county_cdl, load_cdl  # noqa: F401
    from engine.counties import load_counties  # noqa: F401
    from engine.weather.core import fetch_counties_weather, fetch_county_weather  # noqa: F401
