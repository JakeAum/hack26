"""Entry point: ``python -m engine.nass ...`` → :func:`engine.nass.core._main`."""

from __future__ import annotations

from engine.nass.core import _main

if __name__ == "__main__":
    raise SystemExit(_main())
