"""Central logging for the engine — paste-back-friendly cluster output.

Every new module in the engine (``dataset``, ``model``, ``analogs``, ``forecast``)
routes through here so a single AWS run produces:

1. **Pretty stderr** (Rich if installed, plain if not) for live monitoring.
2. **Rotating file sink** under ``~/hack26/data/derived/logs/engine_{ts}.log``
   so every run leaves a self-contained artifact even if the terminal is closed.
3. **Optional second file sink** via ``--log-file PATH`` so you can tag a run
   ("``run_2025.log``") and paste a single file back here for debugging.

Public surface:
    get_logger(name)                 -> logging.Logger
    setup_logging(verbosity, ...)    -> Path  (returns the rotating-file path)
    banner(title, logger=None)       -> None
    log_environment(logger)          -> None
    add_cli_logging_args(parser)     -> None
    apply_cli_logging_args(args)     -> Path

Design rules:

- **No ``print()`` calls in any module that uses this logger.** Every line
  goes through Python ``logging`` so the file sink captures everything you'd
  see on stderr.
- **No hard dependency on Rich.** The forecast extras pull it in for prettier
  output, but the AWS workshop box should never hard-fail on a UI dep.
- **Idempotent.** ``setup_logging`` can be called multiple times; subsequent
  calls only adjust verbosity and add new ``--log-file`` sinks. Existing
  handlers are not duplicated.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Root logger name. Every module-level logger is a child of this so a single
#: ``logging.getLogger("engine").setLevel(...)`` reconfigures everything.
ROOT_LOGGER_NAME = "engine"

#: Maximum size of one rotated log file before it spills to a backup.
_FILE_MAX_BYTES = 50 * 1024 * 1024  # 50 MiB
_FILE_BACKUPS = 5

#: A sentinel attribute we slap on handlers we own so we can identify (and not
#: duplicate) them on repeat ``setup_logging`` calls.
_OWNED_ATTR = "_hack26_owned"

#: Format used by the plain (no-Rich) stderr handler and every file handler.
#: Keeping it consistent so a paste-back from stderr looks the same as the
#: rotated file content.
_FILE_FMT = (
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)-22s  %(message)s"
)
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _data_root() -> Path:
    """Same root convention as the rest of the engine — single place to look
    for everything we write.

    Mirrors ``engine.cdl._data_root`` semantics but does not hard-fail if the
    directory is missing; logging needs to work even before the data root is
    mounted (otherwise the operator never sees the "EFS not mounted" log line
    from CDL). Auto-creates the directory tree we need.
    """
    env = os.environ.get("HACK26_CDL_DATA_DIR") or os.environ.get(
        "HACK26_CACHE_DIR"
    )
    return Path(env) if env else Path.home() / "hack26" / "data"


def _logs_dir() -> Path:
    d = _data_root() / "derived" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_log_path(tag: str = "engine") -> Path:
    """Return the rotated-file path for this run.

    ``tag`` is whatever the entry point wants to call itself — ``train``,
    ``forecast``, ``dataset``. Different CLIs leave distinct files so a
    multi-step pipeline keeps clean separation.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _logs_dir() / f"{tag}_{ts}.log"


# ---------------------------------------------------------------------------
# Handler construction
# ---------------------------------------------------------------------------

def _make_stderr_handler(no_color: bool = False) -> logging.Handler:
    """Pretty Rich handler if available, otherwise stock ``StreamHandler``.

    The plain handler uses the same format as the file sink so a paste-back
    from stderr is visually identical to a paste-back from the log file.
    """
    if not no_color:
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            console = Console(stderr=True, force_terminal=False, soft_wrap=True)
            handler: logging.Handler = RichHandler(
                console=console,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                log_time_format=_DATE_FMT,
            )
            handler.setFormatter(logging.Formatter("%(name)s  %(message)s"))
            return handler
        except Exception:  # noqa: BLE001 - fall back to plain on any import issue
            pass

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    return handler


def _make_file_handler(path: Path, rotating: bool) -> logging.Handler:
    """File sink. Rotating for the default; plain ``FileHandler`` for the
    user-supplied ``--log-file`` so the named file isn't surprise-rotated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if rotating:
        h: logging.Handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=_FILE_MAX_BYTES,
            backupCount=_FILE_BACKUPS,
            encoding="utf-8",
        )
    else:
        h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    return h


def _own(handler: logging.Handler, kind: str) -> logging.Handler:
    """Tag a handler so :func:`setup_logging` can recognize and dedupe it."""
    setattr(handler, _OWNED_ATTR, kind)
    return handler


# ---------------------------------------------------------------------------
# Public: logger factory + one-shot setup
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``engine`` namespace.

    Module-level convention is ``logger = get_logger(__name__)`` at the top
    of every new module. The caller's ``__name__`` ("engine.dataset" etc.)
    is preserved so the format string's ``%(name)s`` shows the source.
    """
    if not name.startswith(ROOT_LOGGER_NAME):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def setup_logging(
    verbosity: int = logging.INFO,
    log_file: Optional[Path] = None,
    no_color: bool = False,
    tag: str = "engine",
) -> Path:
    """Configure the engine root logger. Idempotent.

    Args:
        verbosity: numeric log level (``logging.DEBUG`` / ``INFO`` / ``WARNING``).
        log_file: optional second file sink (in addition to the rotated default).
            Useful for naming a specific run, e.g. ``--log-file run_2025.log``.
        no_color: disable Rich (force plain stderr). Useful when piping to a
            file you'll paste back here, since ANSI escapes are noise.
        tag: filename prefix for the rotated default log
            (``<tag>_<YYYYMMDD_HHMMSS>.log``).

    Returns:
        Path to the rotated default log file (so callers can ``logger.info``
        it on startup; pasting that path back means I can read the full run).
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(verbosity)
    # Don't propagate to Python's root logger — keeps test output clean and
    # avoids double-printing when pytest configures its own handlers.
    root.propagate = False

    # ---- stderr handler (one and only one) -----------------------------------
    stderr_handlers = [
        h for h in root.handlers
        if getattr(h, _OWNED_ATTR, None) == "stderr"
    ]
    if not stderr_handlers:
        root.addHandler(_own(_make_stderr_handler(no_color=no_color), "stderr"))
    else:
        # On a second call, just bring level into sync.
        for h in stderr_handlers:
            h.setLevel(verbosity)

    # ---- rotating-default file handler (also one and only one) --------------
    rotating = [
        h for h in root.handlers
        if getattr(h, _OWNED_ATTR, None) == "rotating"
    ]
    if rotating:
        # Already configured; don't make a second rotating file.
        rotated_path = Path(getattr(rotating[0], "baseFilename", ""))
    else:
        rotated_path = default_log_path(tag=tag)
        h = _make_file_handler(rotated_path, rotating=True)
        h.setLevel(logging.DEBUG)  # file always captures everything
        root.addHandler(_own(h, "rotating"))

    # ---- optional named log file (additive — every call adds a new one) -----
    if log_file is not None:
        log_file = Path(log_file).expanduser().resolve()
        already = any(
            getattr(h, _OWNED_ATTR, None) == "named"
            and Path(getattr(h, "baseFilename", "")) == log_file
            for h in root.handlers
        )
        if not already:
            h = _make_file_handler(log_file, rotating=False)
            h.setLevel(logging.DEBUG)
            root.addHandler(_own(h, "named"))

    # Make sure every level on the root is honored across handlers.
    for h in root.handlers:
        if getattr(h, _OWNED_ATTR, None) == "stderr":
            h.setLevel(verbosity)

    # Quiet noisy upstream loggers that pollute paste-backs.
    for noisy in (
        "urllib3", "urllib3.connectionpool", "requests",
        "fiona", "fiona._env", "rasterio", "rasterio._env",
        "matplotlib", "matplotlib.font_manager", "PIL",
        "stackstac", "pystac_client",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return rotated_path


# ---------------------------------------------------------------------------
# Banners + environment dump
# ---------------------------------------------------------------------------

def banner(
    title: str,
    logger: Optional[logging.Logger] = None,
    char: str = "=",
    width: int = 78,
    level: int = logging.INFO,
) -> None:
    """Log a fenced section header. ``grep`` on the bar line gives you a
    table of contents for the whole run.
    """
    log = logger or get_logger("engine")
    bar = char * width
    ts = datetime.now().strftime(_DATE_FMT)
    pid = os.getpid()
    host = socket.gethostname()
    log.log(level, bar)
    log.log(level, title)
    log.log(level, f"start: {ts}  |  pid: {pid}  |  host: {host}")
    log.log(level, bar)


def _git_sha() -> str:
    """Best-effort git SHA + dirty flag. Returns ``"<no-git>"`` on any failure
    so this never crashes a CLI on a non-git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip())
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "<no-git>"


def _free_disk_gb(path: Path) -> float:
    try:
        import shutil
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except Exception:  # noqa: BLE001
        return float("nan")


def _torch_info() -> dict[str, str]:
    """Soft-import torch so machines without the ``[forecast]`` extra still
    get a clean environment dump."""
    info: dict[str, str] = {"torch": "<not installed>", "cuda": "n/a"}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            dev_count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(dev_count)]
            try:
                vram_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )
            except Exception:  # noqa: BLE001
                vram_gb = float("nan")
            info["cuda"] = (
                f"available={dev_count} device(s); "
                f"primary={names[0]} ({vram_gb} GB)"
                if names else "available=0"
            )
        else:
            info["cuda"] = "torch installed, CUDA unavailable (CPU run)"
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"<import failed: {exc}>"
    return info


def _mask_secret(val: str | None) -> str:
    if not val:
        return "<unset>"
    if len(val) <= 6:
        return "***"
    return f"{val[:3]}***{val[-3:]}  (len={len(val)})"


def log_environment(logger: Optional[logging.Logger] = None) -> None:
    """Dump everything that affects reproducibility into the log on startup.

    Called once at the top of every CLI. Pasting just the environment
    section back lets me debug "why does this work on your laptop but not
    AWS" in a single round trip.
    """
    log = logger or get_logger("engine")
    root = _data_root()
    info = _torch_info()
    log.info("environment:")
    log.info("  python:       %s (%s)",
             platform.python_version(), platform.platform())
    log.info("  torch:        %s", info["torch"])
    log.info("  cuda:         %s", info["cuda"])
    log.info("  data_root:    %s  (free=%s GB)", root, _free_disk_gb(root))
    log.info("  HACK26_CDL_DATA_DIR=%s", os.environ.get("HACK26_CDL_DATA_DIR", "<unset>"))
    log.info("  HACK26_CACHE_DIR=%s",   os.environ.get("HACK26_CACHE_DIR", "<unset>"))
    log.info("  NASS_API_KEY=%s",       _mask_secret(os.environ.get("NASS_API_KEY")))
    log.info("  git:          %s",      _git_sha())
    log.info("  argv:         %s",      " ".join(sys.argv))


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------

def add_cli_logging_args(parser: argparse.ArgumentParser) -> None:
    """Register the universal ``--verbose`` / ``--quiet`` / ``--log-file`` /
    ``--no-color`` flags on a CLI parser. Every new entry point calls this."""
    g = parser.add_argument_group("logging")
    g.add_argument("--verbose", "-v", action="store_true",
                   help="DEBUG-level logging.")
    g.add_argument("--quiet", "-q", action="store_true",
                   help="WARNING+ only. Overrides --verbose.")
    g.add_argument("--log-file", type=Path, default=None,
                   help="Additional log file sink (in addition to the "
                        "rotated default under ~/hack26/data/derived/logs/).")
    g.add_argument("--no-color", action="store_true",
                   help="Disable Rich color output on stderr (cleaner when "
                        "piping to a file you'll paste back).")


def apply_cli_logging_args(
    args: argparse.Namespace, tag: str = "engine"
) -> Path:
    """Apply parsed ``--verbose`` / ``--log-file`` / ``--no-color`` to the
    engine logger and return the rotated-default log path.

    Pair with :func:`add_cli_logging_args`. CLI entry points call this
    immediately after ``parser.parse_args(...)``.
    """
    if getattr(args, "quiet", False):
        level = logging.WARNING
    elif getattr(args, "verbose", False):
        level = logging.DEBUG
    else:
        level = logging.INFO

    return setup_logging(
        verbosity=level,
        log_file=getattr(args, "log_file", None),
        no_color=getattr(args, "no_color", False),
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Lightweight progress logger for long loops
# ---------------------------------------------------------------------------

class StepCounter:
    """Tiny helper for "N/M counties processed" style progress logs.

    Use when looping over many items where you don't want to pull in tqdm.
    Logs at INFO every ``every`` items and on the final item.

    Example::

        sc = StepCounter(logger, total=len(counties), unit="counties", every=25)
        for row in counties.itertuples():
            ...
            sc.tick()
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        unit: str = "items",
        every: int = 25,
        prefix: str = "progress",
    ):
        self._log = logger
        self._total = max(1, int(total))
        self._unit = unit
        self._every = max(1, int(every))
        self._prefix = prefix
        self._n = 0
        self._t0 = datetime.now()

    def tick(self, extra: str | None = None) -> None:
        self._n += 1
        if self._n % self._every == 0 or self._n == self._total:
            elapsed = (datetime.now() - self._t0).total_seconds()
            rate = self._n / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * self._n / self._total
            tail = f"  {extra}" if extra else ""
            self._log.info(
                "%s: %d/%d %s (%.1f%%, %.1fs, %.1f %s/s)%s",
                self._prefix, self._n, self._total, self._unit,
                pct, elapsed, rate, self._unit, tail,
            )
