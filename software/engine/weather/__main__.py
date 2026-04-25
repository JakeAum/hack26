"""Entry point: ``python -m engine.weather ...`` → :func:`engine.weather.core._main`."""

from __future__ import annotations

from engine.weather.core import _main

if __name__ == "__main__":
    raise SystemExit(_main())
