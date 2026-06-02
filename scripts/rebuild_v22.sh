#!/usr/bin/env bash
# v2.2 rebuild — re-extract from CACHED raw MARC (no re-download) with the
# A1-A4 fixes, then match -> sample -> build-v2. Fresh paths keep v2.1 intact.
set -euo pipefail
cd /home/kltang/projects/lcsh-benchmark

DB=data/raw/v2/v2.2.db
CORPUS=data/raw/v2/corpus_v22
MANIFEST=data/raw/v2.2_manifest.json
OUT=data/v2.2

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] === v2.2 rebuild start (branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)) ==="

echo "[$(ts)] STAGE 1/5 acquire-index (columbia+princeton from cache, harvard local)"
uv run lcsh-benchmark-scaleup-run acquire-index \
    --db "$DB" --corpus "$CORPUS" \
    --cache-dir data/raw/v2/raw_cache \
    --sources columbia princeton local

echo "[$(ts)] STAGE 2/5 load"
uv run lcsh-benchmark-scaleup-run load --db "$DB" --corpus "$CORPUS"

echo "[$(ts)] STAGE 3/5 match"
uv run lcsh-benchmark-scaleup-run match --db "$DB"

echo "[$(ts)] STAGE 4/5 sample"
uv run lcsh-benchmark-scaleup-run sample --db "$DB" --out "$MANIFEST"

echo "[$(ts)] STAGE 5/5 build-v2"
uv run lcsh-benchmark-build-v2 --manifest "$MANIFEST" --corpus "$CORPUS" \
    --out "$OUT" --seed 13

echo "[$(ts)] === v2.2 rebuild DONE -> $OUT ==="
