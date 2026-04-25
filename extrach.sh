set -euo pipefail
DIR="${HOME}/hack26/data/derived/forecasts"
shopt -s nullglob
for p in "$DIR"/forecast_2025*.parquet; do
  base=$(basename "$p" .parquet)
  # copy-before-modify: safety snapshot (same bytes as source)
  cp -a -- "$p" "$DIR/${base}.parquet.bak"
  python3 - "$p" "$DIR/${base}.csv" <<'PY'
import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
df.to_csv(sys.argv[2], index=False)
print("wrote", sys.argv[2], "rows=", len(df))
PY
done