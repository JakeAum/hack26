"""USDA NASS Quick Stats — county final yields and state-level forecasts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "nass_api_key",
    "nass_get",
    "fetch_county_nass_yields",
    "fetch_counties_nass_yields",
    "fetch_nass_state_corn_forecasts",
]

_LAZY: dict[str, tuple[str, str]] = {
    "nass_get": ("engine.nass.core", "nass_get"),
    "nass_api_key": ("engine.nass.core", "nass_api_key"),
    "fetch_county_nass_yields": ("engine.nass.core", "fetch_county_nass_yields"),
    "fetch_counties_nass_yields": ("engine.nass.core", "fetch_counties_nass_yields"),
    "fetch_nass_state_corn_forecasts": (
        "engine.nass.core",
        "fetch_nass_state_corn_forecasts",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'engine.nass' has no attribute {name!r}")
    mod_name, attr = target
    import importlib

    return getattr(importlib.import_module(mod_name), attr)


if TYPE_CHECKING:
    from engine.nass.core import (  # noqa: F401
        fetch_counties_nass_yields,
        fetch_county_nass_yields,
        fetch_nass_state_corn_forecasts,
        nass_get,
        nass_api_key,
    )
