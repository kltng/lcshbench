#!/usr/bin/env bash
# te3 retrieval (large + small) on v2.2 dev-2K, reusing the transferred indices.
set -uo pipefail
cd /home/kltang/projects/lcsh-benchmark
set -a; . ./.env; set +a
DEV2K=data/v2.2/dev/dataset_dev_subset2k.json
VOCAB=data/vocab/vocab.jsonl
RUNS=results/runs
ts(){ date '+%H:%M:%S'; }
for variant in large small; do
  name=text-embedding-3-$variant
  O=$RUNS/retrieval_${name}_v22_dev2k.json
  echo "[$(ts)] te3-$variant retrieve (reuse data/index/$name)"
  uv run lcsh-benchmark-retrieve --dataset "$DEV2K" --vocab "$VOCAB" --index-dir data/index \
    --backend openrouter --model "openai/$name" --k 200 --out "$O" || { echo "FAIL retrieve $variant"; continue; }
  echo "[$(ts)] score $variant"
  uv run lcsh-benchmark-score --dataset "$DEV2K" --submission "$O" --out "$RUNS/$(basename "$O" .json).score.json"
  uv run lcsh-benchmark-score-retrieval --dataset "$DEV2K" --submission "$O" --vocab "$VOCAB" \
    --out "$RUNS/$(basename "$O" .json).retrieval-score.json"
done
echo "[$(ts)] te3 done"
