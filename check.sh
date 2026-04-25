#!/usr/bin/env bash
# Pre-flight checks for hack26-forecast on SageMaker / AWS before running
# inference. Mirrors engine path logic (engine/forecast _resolve_model_dir +
# engine/_logging _data_root) so you can see why --model-dir X fails and where
# tft_*.pt actually live.
#
# Usage:
#   ./scripts/forecast_aws_precheck.sh
#   ./scripts/forecast_aws_precheck.sh measurement_5state
#   RUN_FORECAST=1 ./scripts/forecast_aws_precheck.sh measurement_5state   # if resolve succeeds
set -euo pipefail

MODEL_NAME="${1:-measurement_5state}"
export MODEL_NAME

section() { printf '\n==== %s ====\n' "$*"; }

section "Shell / host"
printf 'hostname: %s\n' "$(hostname 2>/dev/null || true)"
printf 'whoami:   %s\n' "$(whoami 2>/dev/null || true)"
printf 'pwd:      %s\n' "$PWD"
printf 'HOME:     %s\n' "${HOME:-}"
printf 'PATH(hack26): which hack26-forecast = %s\n' "$(command -v hack26-forecast 2>/dev/null || echo 'not found')"

section "Relevant environment"
printf 'HACK26_CDL_DATA_DIR: %s\n' "${HACK26_CDL_DATA_DIR:-<unset>}"
printf 'HACK26_CACHE_DIR:     %s\n' "${HACK26_CACHE_DIR:-<unset>}"
printf 'NASS_API_KEY:          %s\n' \
  "$([ -n "${NASS_API_KEY:-}" ] && echo "set (len ${#NASS_API_KEY})" || echo '<unset>')"

section "DATA_ROOT (engine convention, same as engine._logging._data_root)"
if [ -n "${HACK26_CDL_DATA_DIR:-}" ]; then
  DATA_ROOT="${HACK26_CDL_DATA_DIR}"
elif [ -n "${HACK26_CACHE_DIR:-}" ]; then
  DATA_ROOT="${HACK26_CACHE_DIR}"
else
  DATA_ROOT="${HOME}/hack26/data"
fi
export DATA_ROOT
printf 'DATA_ROOT=%s\n' "$DATA_ROOT"
if [ -e "$DATA_ROOT" ]; then
  printf 'resolved: %s\n' "$(readlink -f "$DATA_ROOT" 2>/dev/null || realpath "$DATA_ROOT" 2>/dev/null || echo "$DATA_ROOT")"
  printf 'is_dir:   %s\n' "$([ -d "$DATA_ROOT" ] && echo yes || echo no)"
else
  printf 'ERROR: DATA_ROOT does not exist — engine cannot use derived/models here.\n'
fi

section "Model-dir resolution (what hack26-forecast does for argument: %s)" "$MODEL_NAME"
# 1) Relative path as resolved from CWD (Path(...).resolve())
TRY_CWD="${PWD}/${MODEL_NAME}"
printf 'Try 1 — cwd-relative resolve (ERROR message shows this if both fail): %s\n' "$TRY_CWD"
if [ -d "$TRY_CWD" ]; then
  echo "  -> EXISTS (this directory would be used)"
else
  echo "  -> missing (expected in repo root unless you copied checkpoints here)"
fi
# 2) Single-part name -> data_root/derived/models/<name>
TRY_ALT="${DATA_ROOT}/derived/models/${MODEL_NAME}"
printf 'Try 2 — DATA_ROOT/derived/models/%s: %s\n' "$MODEL_NAME" "$TRY_ALT"
if [ -d "$TRY_ALT" ]; then
  echo "  -> EXISTS (inference will use this if Try 1 is missing — must match training output dir)"
else
  echo "  -> missing (training must have written checkpoints somewhere else, or this run has no EFS data)"
fi

section "Contents: DATA_ROOT/derived/models (if present)"
if [ -d "${DATA_ROOT}/derived/models" ]; then
  ls -la "${DATA_ROOT}/derived/models" || true
else
  echo "No directory: ${DATA_ROOT}/derived/models"
fi

section "Search for checkpoint files (tft_*.pt) under derived/models (max depth 3)"
if [ -d "${DATA_ROOT}/derived" ]; then
  # shellcheck disable=SC2012
  find "${DATA_ROOT}/derived/models" -maxdepth 3 -name 'tft_*.pt' -type f 2>/dev/null | head -200 || true
  COUNT="$(find "${DATA_ROOT}/derived/models" -maxdepth 3 -name 'tft_*.pt' -type f 2>/dev/null | wc -l | tr -d ' ')"
  printf 'Total tft_*.pt found (capped list above): %s\n' "${COUNT:-0}"
else
  echo "Skip find: ${DATA_ROOT}/derived not found"
fi

section "Python: exact engine _resolve_model_dir logic"
python3 -u << 'PY'
from __future__ import annotations
import os
import sys
from pathlib import Path

name = os.environ["MODEL_NAME"]

def _data_root() -> Path:
    env = os.environ.get("HACK26_CDL_DATA_DIR") or os.environ.get("HACK26_CACHE_DIR")
    return Path(env) if env else Path.home() / "hack26" / "data"

def resolve_model_dir(model_dir: str) -> Path:
    raw = Path(model_dir)
    expanded = raw.expanduser()
    p = expanded.resolve()
    if p.exists():
        return p
    alt: Path | None = None
    if not expanded.is_absolute() and len(expanded.parts) == 1:
        alt = (_data_root() / "derived" / "models" / expanded.parts[0]).resolve()
        if alt.exists():
            return alt
    raise FileNotFoundError(
        f"--model-dir does not exist: {p}\n"
        f"  (tried data_root/derived/models/<name> -> {alt!s})"
    )

print("_data_root():", _data_root())
print("len(Path(%r).parts):" % name, len(Path(name).parts), Path(name).parts)
try:
    p = resolve_model_dir(name)
    print("RESOLVED --model-dir ->", p)
    sys.exit(0)
except FileNotFoundError as e:
    print(str(e), file=sys.stderr)
    # Still show parts debug for the failure case
    ex = Path(name).expanduser()
    print("debug: expanded.is_absolute() =", ex.is_absolute(), "parts =", ex.parts, file=sys.stderr)
    sys.exit(1)
PY
PY_EXIT=$?
if [ "$PY_EXIT" -ne 0 ]; then
  section "Summary"
  echo "Python resolution FAILED — hack26-forecast will fail until checkpoints exist in Try 1 or Try 2."
  echo "Fix: pass an absolute path to the directory that contains tft_aug1.pt, tft_sep1.pt, tft_oct1.pt, tft_final.pt"
  echo "     e.g.  --model-dir /path/to/that/directory"
  exit "$PY_EXIT"
fi

section "Suggested hack26-forecast line (use absolute --model-dir from RESOLVED line above)"
RESOLVED_FOR_CMD="$(python3 -c "
import os
from pathlib import Path
name = os.environ['MODEL_NAME']
def _dr():
    e = os.environ.get('HACK26_CDL_DATA_DIR') or os.environ.get('HACK26_CACHE_DIR')
    return Path(e) if e else Path.home() / 'hack26' / 'data'
def go():
    ex = Path(name).expanduser()
    p = ex.resolve()
    if p.exists():
        return p
    if not ex.is_absolute() and len(ex.parts) == 1:
        a = (_dr() / 'derived' / 'models' / ex.parts[0]).resolve()
        if a.exists():
            return a
    return None
r = go()
print(r) if r else print('')
")"
if [ -n "$RESOLVED_FOR_CMD" ]; then
  cat << CMD

hack26-forecast \\
  --year 2025 --all-dates \\
  --states Iowa Colorado Wisconsin Missouri Nebraska \\
  --model-dir ${RESOLVED_FOR_CMD} \\
  --history-start 2008 --history-end 2024 \\
  --num-samples 500 --allow-download \\
  --out \${HOME}/hack26/data/derived/forecasts/forecast_2025.parquet \\
  --max-fetch-workers 8 \\
  -v --log-file \${HOME}/hack26/data/derived/logs/forecast_2025.log
CMD
fi

if [ "${RUN_FORECAST:-0}" = "1" ] && [ -n "${RESOLVED_FOR_CMD:-}" ]; then
  section "RUN_FORECAST=1 — running hack26-forecast"
  exec hack26-forecast \
    --year 2025 --all-dates \
    --states Iowa Colorado Wisconsin Missouri Nebraska \
    --model-dir "$RESOLVED_FOR_CMD" \
    --history-start 2008 --history-end 2024 \
    --num-samples 500 --allow-download \
    --out "${HOME}/hack26/data/derived/forecasts/forecast_2025.parquet" \
    --max-fetch-workers 8 \
    -v --log-file "${HOME}/hack26/data/derived/logs/forecast_2025.log"
fi

section "Done"
printf 'To run the forecast after verifying paths: RUN_FORECAST=1 %s %s\n' "$0" "$MODEL_NAME"
exit 0
