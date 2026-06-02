#!/usr/bin/env bash
set -uo pipefail
cd /home/kltang/projects/lcsh-benchmark
set -a; . ./.env; set +a
echo "[wait] te3-large build (poll progress.json for complete:true)"
until uv run python -c "import json,sys;sys.exit(0 if json.load(open('data/index/text-embedding-3-large/progress.json'))['complete'] else 1)" 2>/dev/null; do
  sleep 90
done
echo "[wait] te3-large index COMPLETE — uploading to HF"
uv run python -c "
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(repo_id='kltng/lcsh-bench-v22-indices',repo_type='dataset',
    folder_path='data/index/text-embedding-3-large',path_in_repo='text-embedding-3-large',
    commit_message='index text-embedding-3-large')
print('te3-large uploaded')
"
echo "[wait] READY — te3-large on HF; lean job can run"
