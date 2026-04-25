"""Engine: pluggable data-source layer.

Every source is a function that takes a county (geoid + geometry) and returns
a tidy DataFrame. The County Catalog (`engine.counties`) is the single source
of truth for ROIs and geometries; all other sources join on `geoid`.
"""

from engine.counties import load_counties

__all__ = ["load_counties"]
