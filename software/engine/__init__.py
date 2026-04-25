"""Engine: pluggable data-source layer.

Every source is a function that takes a county (geoid + geometry) and returns
a tidy DataFrame. The County Catalog (`engine.counties`) is the single source
of truth for ROIs and geometries; all other sources join on `geoid`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["load_counties"]

# Lazy re-export so `python -m engine.counties` doesn't double-import the
# submodule (which triggers a runpy RuntimeWarning).
def __getattr__(name: str) -> Any:
    if name == "load_counties":
        from engine.counties import load_counties
        return load_counties
    raise AttributeError(f"module 'engine' has no attribute {name!r}")


if TYPE_CHECKING:
    from engine.counties import load_counties  # noqa: F401
