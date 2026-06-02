#!/usr/bin/env bash
# Pull lcsh.db + FT embedder from HF (reliable HTTPS, unlike the flaky Tailscale link).
set -uo pipefail
cd /home/kltang/projects/lcsh-benchmark
set -a; . ./.env; set +a
ts(){ date '+%H:%M:%S'; }
echo "[$(ts)] pulling kltng/lcsh-db -> data/lcsh.db (4.35 GB)"
echo "[$(ts)] pulling kltng/embeddinggemma-300m-lcsh -> data/finetune/embeddinggemma-300m-lcsh-merged (56 MB)"
uv run python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download, snapshot_download
tok=os.environ["HF_TOKEN"]
# lcsh.db (authority DB)
p=hf_hub_download("kltng/lcsh-db","lcsh.db",repo_type="dataset",token=tok)
os.makedirs("data",exist_ok=True); shutil.copy(p,"data/lcsh.db")
print("lcsh.db ->", os.path.getsize("data/lcsh.db")/1e9, "GB")
# FT embedder: local dir named to MATCH the existing index backend.name
#   index = embeddinggemma-300m-lcsh-merged-d256  ==>  model dir basename must be ...-lcsh-merged
d=snapshot_download("kltng/embeddinggemma-300m-lcsh",token=tok,
                    local_dir="data/finetune/embeddinggemma-300m-lcsh-merged")
print("FT model ->", d)
PY
echo "[$(ts)] hf_pull done"
