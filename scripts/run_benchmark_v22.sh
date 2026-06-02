#!/usr/bin/env bash
# Turnkey v2.2 benchmark run. Runs every system whose prerequisites (keys,
# packages, indices, models, lcsh.db) are present; SKIPS the rest with a logged
# reason — never silently. Data-only parts (stats/Table 1, leakage, frequency,
# scoring) always run. Safe to re-run: existing submissions are reused.
#
# Prereqs by system (set up as they arrive):
#   te3          -> openrouter_api_key (.env)            [+ data/index/text-embedding-3-small to reuse]
#   gemma stock  -> uv sync --extra local-embed, HF_TOKEN [+ data/index/embeddinggemma-300m-d256]
#   gemma FT     -> local-embed + $FT_MODEL              [+ data/index/embeddinggemma-300m-lcsh-merged-d256]
#   lcsh-onnx    -> lcsh_db_builder importable + $LCSH_DB (+ onnxruntime)
#   gen/sel/llmrr-> openrouter_api_key (or deepseek_api_key)
#   bge rerank   -> local-embed
set -uo pipefail   # NOT -e: keep going past a skipped/failed system; report at end
cd /home/kltang/projects/lcsh-benchmark

# ---------- CONFIG (verify index dir names vs transferred data/index/* once assets land) ----------
DATA=data/v2.2
DEV=$DATA/dev/dataset_dev.json
DEV2K=$DATA/dev/dataset_dev_subset2k.json
TEST=$DATA/test/dataset_test.json
HASHED=$DATA/test/gt_test.hashed.json
VOCAB=data/vocab/vocab.jsonl
INDEX=data/index
CORPUS=data/raw/v2/corpus_v22
RUNS=results/runs
STATS_OUT=results/dataset-stats-v22.md
BOARD=results/leaderboard-v22.md

TE3_MODEL=openai/text-embedding-3-large               # auto-overridden below to the index that exists
STOCK_MODEL=google/embeddinggemma-300m                # local d256 -> index: embeddinggemma-300m-d256
FT_MODEL=data/finetune/embeddinggemma-300m-lcsh-merged # local d256 -> index: ...-lcsh-merged-d256
LCSH_DB=data/lcsh.db
CHAT_PROVIDER=openrouter
CHAT_MODEL=deepseek/deepseek-chat                     # TODO: confirm exact OpenRouter slug for deepseek-v4-flash
RERANK_BGE=BAAI/bge-reranker-v2-m3

mkdir -p "$RUNS" results
PRED_SUBS=()        # prediction submissions to score + leaderboard
SKIPPED=()

# ---------- helpers ----------
ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
skip(){ SKIPPED+=("$1 — $2"); log "SKIP  $1: $2"; }
have_key(){ [ -n "${!1:-}" ]; }
have_path(){ [ -e "$1" ] && [ -s "$1" ] 2>/dev/null || [ -d "$1" ]; }
have_pkg(){ uv run python -c "import $1" 2>/dev/null; }
emit(){ PRED_SUBS+=("$1"); }   # register a prediction submission

# load .env (keys) into the environment
if [ -f .env ]; then set -a; . ./.env; set +a; log "loaded .env"; else log "no .env (API systems will skip)"; fi

# preflight: dataset must exist (rebuild done)
have_path "$DEV" || { echo "FATAL: $DEV missing — run the v2.2 rebuild first."; exit 1; }

# ---------- Phase 0: dev-2K subset ----------
if have_path "$DEV2K"; then log "dev-2K subset present"; else
  log "build dev-2K subset"; uv run lcsh-benchmark-subset --dataset "$DEV" --out "$DEV2K" --target 2000 --seed 13
fi

# ---------- Phase 1: data-only (always) ----------
log "STATS / Table 1 -> $STATS_OUT"
uv run lcsh-benchmark-stats --indices "$CORPUS"/*/*.jsonl --dataset "$DEV" --vocab "$VOCAB" --out "$STATS_OUT" \
  || log "WARN stats failed (continuing)"
log "leakage audit -> results/leakage-v22.txt"
uv run lcsh-benchmark-audit-leakage --dataset "$DEV" --vocab "$VOCAB" | tee results/leakage-v22.txt \
  || log "WARN leakage audit failed"
log "frequency floor -> dev2k"
F=$RUNS/frequency_v22_dev2k.json
uv run lcsh-benchmark-baseline-frequency --dataset "$DEV2K" --freq-from "$DEV" --out "$F" && emit "$F" \
  || log "WARN frequency failed"

# ---------- Phase 2: neural / API (guarded) ----------
# te3 (OpenRouter) — auto-pick whichever index is present (large preferred); never rebuild blind
for variant in large small; do
  if have_path "$INDEX/text-embedding-3-$variant"; then TE3_MODEL="openai/text-embedding-3-$variant"; break; fi
done
TE3_NAME=${TE3_MODEL##*/}                              # e.g. text-embedding-3-large
if have_key openrouter_api_key; then
  if have_path "$INDEX/$TE3_NAME"; then
    O=$RUNS/retrieval_${TE3_NAME}_v22_dev2k.json
    log "te3 retrieve ($TE3_MODEL, reusing $INDEX/$TE3_NAME)"
    uv run lcsh-benchmark-retrieve --dataset "$DEV2K" --vocab "$VOCAB" --index-dir "$INDEX" \
      --backend openrouter --model "$TE3_MODEL" --k 200 --out "$O" && emit "$O" || log "WARN te3 failed"
  else skip "te3" "no $INDEX/$TE3_NAME index — refusing to rebuild 515K embeddings via paid API"; fi
else skip "te3" "openrouter_api_key not set"; fi

# stock EmbeddingGemma (local)
if have_pkg sentence_transformers; then
  O=$RUNS/retrieval_embgemma-stock_v22_dev2k.json
  log "gemma stock retrieve (index embeddinggemma-300m-d256)"
  uv run lcsh-benchmark-retrieve --dataset "$DEV2K" --vocab "$VOCAB" --index-dir "$INDEX" \
    --backend local --model "$STOCK_MODEL" --truncate-dim 256 --k 200 --out "$O" && emit "$O" || log "WARN stock gemma failed"
else skip "embgemma-stock" "sentence_transformers missing (uv sync --extra local-embed)"; fi

# fine-tuned EmbeddingGemma (local)
if have_pkg sentence_transformers && have_path "$FT_MODEL"; then
  O=$RUNS/retrieval_embgemma-ft-clean_v22_dev2k.json
  log "gemma FT retrieve (index embeddinggemma-300m-lcsh-merged-d256)"
  uv run lcsh-benchmark-retrieve --dataset "$DEV2K" --vocab "$VOCAB" --index-dir "$INDEX" \
    --backend local --model "$FT_MODEL" --truncate-dim 256 --k 200 --out "$O" && emit "$O" || log "WARN FT gemma failed"
else skip "embgemma-ft-clean" "need sentence_transformers + $FT_MODEL"; fi

# lcsh-onnx (raw + vernacular) — fully-local ONNX pipeline; needs lcsh_db_builder + lcsh.db
if have_pkg lcsh_db_builder && have_path "$LCSH_DB"; then
  for V in "" "--vernacular"; do
    tag=$([ -n "$V" ] && echo vern || echo raw)
    O=$RUNS/lcsh_onnx_${tag}_v22_dev2k.json
    log "lcsh-onnx $tag retrieve+rerank (CPU onnxruntime)"
    uv run python src/lcsh_benchmark/baselines/lcsh_onnx_adapter.py \
      --db "$LCSH_DB" --dataset "$DEV2K" --out "$O" --pool 200 $V && emit "$O" || log "WARN lcsh-onnx $tag failed"
  done
else skip "lcsh-onnx" "need lcsh_db_builder importable + $LCSH_DB (pull from HF private repo; uv add onnxruntime)"; fi

# bge rerank over te3 L1 (local)
TE3_SUB=$RUNS/retrieval_te3-small_v22_dev2k.json
if have_pkg sentence_transformers && have_path "$TE3_SUB"; then
  O=$RUNS/rerank_bge_te3_v22_dev2k.json
  log "bge rerank over te3"
  uv run lcsh-benchmark-rerank --dataset "$DEV2K" --l1-submission "$TE3_SUB" \
    --backend local --model "$RERANK_BGE" --top-n 50 --out "$O" && emit "$O" || log "WARN bge rerank failed"
else skip "bge-rerank" "need sentence_transformers + te3 submission"; fi

# DeepSeek llm-rerank over te3 L1, and selection, and generation
if have_key openrouter_api_key || have_key deepseek_api_key; then
  if have_path "$TE3_SUB"; then
    O=$RUNS/llmrr_${CHAT_PROVIDER}_te3_v22_dev2k.json
    log "llm-rerank ($CHAT_MODEL) over te3"
    uv run lcsh-benchmark-llm-rerank --dataset "$DEV2K" --l1-submission "$TE3_SUB" \
      --provider "$CHAT_PROVIDER" --model "$CHAT_MODEL" --top-n 50 --out "$O" && emit "$O" || log "WARN llm-rerank failed"
    S=$RUNS/sel_${CHAT_PROVIDER}_v22_dev2k.json
    log "selection ($CHAT_MODEL) over te3"
    uv run lcsh-benchmark-select --dataset "$DEV2K" --l1-submission "$TE3_SUB" \
      --provider "$CHAT_PROVIDER" --model "$CHAT_MODEL" --top-n 50 --out "$S" && emit "$S" || log "WARN selection failed"
  else skip "llm-rerank/selection" "no te3 L1 submission to rerank"; fi
  G=$RUNS/gen_${CHAT_PROVIDER}_v22_dev2k.json
  log "generation ($CHAT_MODEL)"
  uv run lcsh-benchmark-generate --dataset "$DEV2K" --provider "$CHAT_PROVIDER" --model "$CHAT_MODEL" \
    --out "$G" && emit "$G" || log "WARN generation failed"
else skip "gen/sel/llm-rerank" "no openrouter_api_key/deepseek_api_key"; fi

# ---------- Phase 3: score everything present + leaderboard ----------
log "SCORING ${#PRED_SUBS[@]} submissions"
for sub in "${PRED_SUBS[@]}"; do
  [ -f "$sub" ] || continue
  base=$(basename "$sub" .json)
  uv run lcsh-benchmark-score --dataset "$DEV2K" --submission "$sub" --out "$RUNS/${base}.score.json" \
    || log "WARN score failed: $base"
  # retrieval-style submissions also get reachable-GT recall@k
  if [[ "$base" == retrieval_* || "$base" == lcsh_onnx_* || "$base" == frequency_* ]]; then
    uv run lcsh-benchmark-score-retrieval --dataset "$DEV2K" --submission "$sub" --vocab "$VOCAB" \
      --out "$RUNS/${base}.retrieval-score.json" || log "WARN score-retrieval failed: $base"
  fi
done

if [ ${#PRED_SUBS[@]} -gt 0 ]; then
  log "leaderboard -> $BOARD"
  uv run lcsh-benchmark-leaderboard --dataset "$DEV2K" --submissions "${PRED_SUBS[@]}" --out "$BOARD" \
    || log "WARN leaderboard failed"
fi

# ---------- summary ----------
echo; echo "==================== v2.2 benchmark summary ===================="
echo "scored submissions: ${#PRED_SUBS[@]}"; printf '  - %s\n' "${PRED_SUBS[@]}"
echo "Table 1: $STATS_OUT   leaderboard: $BOARD"
if [ ${#SKIPPED[@]} -gt 0 ]; then echo "SKIPPED (missing prereqs):"; printf '  - %s\n' "${SKIPPED[@]}"; fi
echo "================================================================"
