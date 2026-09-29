#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADAPTER="${1:-${TAIJI_ADAPTER:-$ROOT/Taiji-2B}}"
BASE="${TAIJI_BASE_MODEL:-Qwen/Qwen3.5-2B}"
if [[ ! -d "$ADAPTER" ]]; then
  echo "Adapter directory not found: $ADAPTER" >&2
  echo "Download it with: hf download nigelleong0703/Taiji-2B --local-dir Taiji-2B" >&2
  exit 2
fi
OUTPUT="$(mktemp "${TMPDIR:-/tmp}/taiji-demo.XXXXXX.jsonl")"
trap 'rm -f "$OUTPUT"' EXIT
python "$ROOT/inference/infer.py" --base "$BASE" --adapter "$ADAPTER" \
  --data "$ROOT/examples/requests.jsonl" --output "$OUTPUT"
cat "$OUTPUT"
