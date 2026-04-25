"""Smoke tests for :mod:`engine._logging` — confirm the central logger
configures handlers, writes a file, and the helpers don't crash.

All tests here are offline and finish in well under a second.

    pytest software/tests/test_logging_smoke.py -q
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pytest

from engine import _logging as elog


@pytest.fixture(autouse=True)
def _isolate_engine_logger():
    """Tear down the engine root logger between tests so handler counts and
    file paths don't leak across cases."""
    root = logging.getLogger(elog.ROOT_LOGGER_NAME)
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_propagate = root.propagate
    root.handlers.clear()
    yield
    root.handlers.clear()
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)
    root.propagate = saved_propagate


def _redirect_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HACK26_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HACK26_CDL_DATA_DIR", raising=False)


def test_setup_logging_creates_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    log_path = elog.setup_logging(verbosity=logging.INFO, tag="testrun")
    log = elog.get_logger("test")
    log.info("hello world")
    log.warning("warning line")

    # Force the file handler to flush.
    for h in logging.getLogger(elog.ROOT_LOGGER_NAME).handlers:
        h.flush()

    assert log_path.exists(), f"expected rotated log file at {log_path}"
    body = log_path.read_text(encoding="utf-8")
    assert "hello world" in body
    assert "warning line" in body
    assert log_path.parent == tmp_path / "derived" / "logs"


def test_setup_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    elog.setup_logging(verbosity=logging.DEBUG, tag="idem")
    n_handlers_first = len(logging.getLogger(elog.ROOT_LOGGER_NAME).handlers)
    elog.setup_logging(verbosity=logging.WARNING, tag="idem")
    n_handlers_second = len(logging.getLogger(elog.ROOT_LOGGER_NAME).handlers)
    assert n_handlers_second == n_handlers_first, (
        "setup_logging should not duplicate handlers on repeat calls"
    )
    # And the verbosity should be honored on the second call.
    assert logging.getLogger(elog.ROOT_LOGGER_NAME).level == logging.WARNING


def test_log_file_arg_adds_named_sink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    extra = tmp_path / "extra.log"
    elog.setup_logging(verbosity=logging.INFO, log_file=extra, tag="named")
    elog.get_logger("named").info("named-sink message")

    for h in logging.getLogger(elog.ROOT_LOGGER_NAME).handlers:
        h.flush()
    assert extra.exists()
    assert "named-sink message" in extra.read_text(encoding="utf-8")


def test_banner_and_get_logger_dont_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    elog.setup_logging(verbosity=logging.INFO, tag="banner")
    log = elog.get_logger("engine.test.banner")
    elog.banner("STEP 1/2  Smoke test of banner", logger=log)
    elog.banner("STEP 2/2  Done")  # also OK without explicit logger


def test_log_environment_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    monkeypatch.setenv("NASS_API_KEY", "abcdef1234567890")
    elog.setup_logging(verbosity=logging.INFO, tag="env")
    # Should never raise even on a machine without torch.
    elog.log_environment()
    rotated = elog.default_log_path("env")
    # Rotated path now in the tmpdir (created earlier; not guaranteed to be
    # the SAME timestamp as the one above, so just verify the parent dir).
    assert rotated.parent == tmp_path / "derived" / "logs"


def test_step_counter_emits_at_intervals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The engine logger has propagate=False (so its rotating file sink is the
    # source of truth, not pytest's `caplog` which reads off the root). Read
    # the rotating file we configured.
    _redirect_data_root(monkeypatch, tmp_path)
    log_path = elog.setup_logging(verbosity=logging.INFO, tag="step")
    log = elog.get_logger("step")
    sc = elog.StepCounter(log, total=5, unit="things", every=2, prefix="probe")
    for _ in range(5):
        sc.tick()
    for h in logging.getLogger(elog.ROOT_LOGGER_NAME).handlers:
        h.flush()

    body = log_path.read_text(encoding="utf-8")
    assert "probe: 2/5 things" in body
    assert "probe: 4/5 things" in body
    # Last tick (5/5) must always emit.
    assert "probe: 5/5 things" in body


def test_cli_argparse_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    parser = argparse.ArgumentParser()
    elog.add_cli_logging_args(parser)
    args = parser.parse_args(["--verbose", "--no-color",
                              "--log-file", str(tmp_path / "cli.log")])
    log_path = elog.apply_cli_logging_args(args, tag="cli")
    elog.get_logger("cli").debug("debug should be captured at -v level")

    for h in logging.getLogger(elog.ROOT_LOGGER_NAME).handlers:
        h.flush()
    assert log_path.exists()
    assert (tmp_path / "cli.log").exists()
    txt = (tmp_path / "cli.log").read_text(encoding="utf-8")
    assert "debug should be captured" in txt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
