# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub>=0.26", "numpy", "httpx", "tqdm",
#   "sentence-transformers>=3.3", "peft>=0.7", "einops",
#   "onnxruntime-gpu>=1.20", "transformers>=4.45", "torch>=2.5",
#   "sqlite-vec>=0.1.5", "pymarc>=5.3", "requests>=2.32",
# ]
# ///
"""v2.2 benchmark on HF Jobs. Pulls code+data from HF, builds indices on GPU,
runs every system (te3 / stock+FT gemma / lcsh-onnx / deepseek gen+sel),
scores, and pushes results to an HF dataset repo. Resilient: each system is
guarded and failures are logged, not fatal; results upload at the end.

Secrets expected in env: openrouter_api_key, deepseek_api_key, HF_TOKEN
Env knobs: SMOKE=1 (setup + frequency only, for a cheap CPU validation run).
"""
import os, sys, subprocess, json, time
from pathlib import Path

TOK = os.environ["HF_TOKEN"]
SMOKE = os.environ.get("SMOKE") == "1"
BUNDLE = "kltng/lcsh-bench-v22-job"
RESULTS_REPO = "kltng/lcsh-bench-v22-results"
WORK = Path("/tmp/lcsh"); WORK.mkdir(exist_ok=True)
PY = sys.executable
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

# ---------------- 1. fetch code + data from HF ----------------
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
log("downloading bundle (code + v2.2 data)")
bundle = Path(snapshot_download(BUNDLE, repo_type="dataset", token=TOK, local_dir=str(WORK/"bundle")))
log("downloading vocab")
vocab = hf_hub_download("kltng/lcsh-finetune", "vocab.jsonl", repo_type="dataset", token=TOK)
lcsh_db = ft_model = None
if not SMOKE:
    log("downloading FT model")
    # name the FT dir so backend.name == the index name convention (...-lcsh-merged-d256)
    ft_model = snapshot_download("kltng/embeddinggemma-300m-lcsh", token=TOK,
                                 local_dir=str(WORK/"embeddinggemma-300m-lcsh-merged"))
    if os.environ.get("FULL") == "1":   # lcsh.db (4.35 GB) only needed by lcsh-onnx
        log("downloading lcsh.db (FULL)")
        lcsh_db = hf_hub_download("kltng/lcsh-db", "lcsh.db", repo_type="dataset", token=TOK)

# ---------------- 2. install the two private packages ----------------
log("uv pip install lcsh-benchmark + lcsh_db_builder")
subprocess.run(["uv", "pip", "install", "--python", PY, "-q", "--no-deps",
                str(bundle/"lcsh-benchmark"), str(bundle/"db-builder")], check=True)
BIN = Path(PY).parent           # console scripts live here after install
def cli(name): return str(BIN/name)

# ---------------- 3. layout + env ----------------
DATA = bundle/"data"/"v2.2"
DEV2K = DATA/"dev"/"dataset_dev_subset2k.json"
DEV = DATA/"dev"/"dataset_dev.json"
INDEX = WORK/"index"; INDEX.mkdir(exist_ok=True)
if not SMOKE:   # reuse the prebuilt te3 indices from HF -> retrieve skips the API rebuild
    for name in ("text-embedding-3-large", "text-embedding-3-small",
                 "embeddinggemma-300m-lcsh-merged-d256"):   # reuse FT index -> no GPU build, no OOM
        try:
            snapshot_download("kltng/lcsh-bench-v22-indices", repo_type="dataset", token=TOK,
                              allow_patterns=[f"{name}/*"], local_dir=str(INDEX))
            log(f"reused index {name}")
        except Exception as e:
            log(f"WARN could not fetch index {name} ({e}); retrieve will rebuild via API")
RUNS = WORK/"runs"; RUNS.mkdir(exist_ok=True)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device={DEVICE}  dev2k={DEV2K.exists()}")
env = dict(os.environ)
preds = []   # submission paths
skipped = []

def run(cmd, label):
    log("RUN", label)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0: log("WARN failed:", label); return False
    return True

def have_key(k): return bool(os.environ.get(k))

# ---------------- 4. frequency floor (always; smoke stops here) ----------------
F = RUNS/"frequency_v22_dev2k.json"
if run([cli("lcsh-benchmark-baseline-frequency"), "--dataset", str(DEV2K),
        "--freq-from", str(DEV), "--out", str(F)], "frequency"):
    preds.append(F)

if SMOKE:
    log("SMOKE done — setup + frequency validated");
else:
  # ---------------- 5. te3 large + small (API; builds index over vocab) ----------------
  if have_key("openrouter_api_key"):
    for variant in ("large", "small"):
        name = f"text-embedding-3-{variant}"
        O = RUNS/f"retrieval_{name}_v22_dev2k.json"
        if run([cli("lcsh-benchmark-retrieve"), "--dataset", str(DEV2K), "--vocab", vocab,
                "--index-dir", str(INDEX), "--backend", "openrouter", "--model", f"openai/{name}",
                "--k", "200", "--out", str(O), "--max-cost", "5.0"], f"te3-{variant}"):
            preds.append(O)
  else: skipped.append("te3 (no openrouter_api_key)")

  # ---------------- 6. local embedders: FT reuses its index (query-embed only); stock builds (FULL only) ----------------
  FULL = os.environ.get("FULL") == "1"
  # stock builds fast on GPU (batch 32); lcsh-onnx stays FULL-only (needs onnxruntime-gpu CUDA libs)
  gemma = [("ft", ft_model, "embeddinggemma-300m-lcsh-merged-d256"),
           ("stock", "google/embeddinggemma-300m", "embeddinggemma-300m-d256")]
  for label, model, idxname in gemma:
    O = RUNS/f"retrieval_embgemma-{label}_v22_dev2k.json"
    if run([cli("lcsh-benchmark-retrieve"), "--dataset", str(DEV2K), "--vocab", vocab,
            "--index-dir", str(INDEX), "--backend", "local", "--model", model,
            "--truncate-dim", "256", "--device", DEVICE, "--k", "200", "--out", str(O),
            "--batch-size", "32"], f"gemma-{label}"):   # small batch -> no GPU OOM on the stock build
        preds.append(O)

  # ---------------- 7. deepseek gen + sel ----------------
  if have_key("openrouter_api_key") or have_key("deepseek_api_key"):
    prov = "deepseek" if have_key("deepseek_api_key") else "openrouter"
    model = "deepseek-chat" if prov == "deepseek" else "deepseek/deepseek-chat"
    G = RUNS/f"gen_{prov}_v22_dev2k.json"
    if run([cli("lcsh-benchmark-generate"), "--dataset", str(DEV2K), "--provider", prov,
            "--model", model, "--out", str(G), "--max-cost", "5.0"], "gen"): preds.append(G)
    te3 = RUNS/"retrieval_text-embedding-3-large_v22_dev2k.json"
    if te3.exists():
        S = RUNS/f"sel_{prov}_v22_dev2k.json"
        if run([cli("lcsh-benchmark-select"), "--dataset", str(DEV2K), "--l1-submission", str(te3),
                "--provider", prov, "--model", model, "--top-n", "50", "--out", str(S),
                "--max-cost", "5.0"], "sel"): preds.append(S)
  else: skipped.append("gen/sel (no chat key)")

  # ---------------- 8. checkpoint upload — core results safe BEFORE the risky lcsh-onnx ----------------
  if FULL and preds:
    try:
        _api = HfApi(token=TOK); _api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
        _api.upload_folder(repo_id=RESULTS_REPO, repo_type="dataset", folder_path=str(RUNS),
                           path_in_repo="runs", commit_message="checkpoint before lcsh-onnx")
        log("checkpoint uploaded — core submissions safe")
    except Exception as e: log("WARN checkpoint upload failed:", e)

  # ---------------- 9. lcsh-onnx (raw + vernacular) — LAST: slow even on GPU, may time out ----------------
  adapter = bundle/"lcsh-benchmark"/"src"/"lcsh_benchmark"/"baselines"/"lcsh_onnx_adapter.py"
  for tag, extra in ((("raw", []), ("vern", ["--vernacular"])) if FULL else ()):
    O = RUNS/f"lcsh_onnx_{tag}_v22_dev2k.json"
    if run([PY, str(adapter), "--db", lcsh_db, "--dataset", str(DEV2K),
            "--out", str(O), "--pool", "200"] + extra, f"lcsh-onnx-{tag}"):
        preds.append(O)

# ---------------- 9. score everything + leaderboard ----------------
log(f"scoring {len(preds)} submissions")
for sub in preds:
    base = sub.stem
    run([cli("lcsh-benchmark-score"), "--dataset", str(DEV2K), "--submission", str(sub),
         "--out", str(RUNS/f"{base}.score.json")], f"score {base}")
    if base.startswith(("retrieval_", "lcsh_onnx_", "frequency_")):
        run([cli("lcsh-benchmark-score-retrieval"), "--dataset", str(DEV2K), "--submission", str(sub),
             "--vocab", vocab, "--out", str(RUNS/f"{base}.retrieval-score.json")], f"score-retrieval {base}")
BOARD = RUNS/"leaderboard-v22.md"
if preds:
    run([cli("lcsh-benchmark-leaderboard"), "--dataset", str(DEV2K),
         "--submissions", *map(str, preds), "--out", str(BOARD)], "leaderboard")

# stats (Table 1) — composition only (funnel needs corpus shards, not on HF)
run([cli("lcsh-benchmark-stats"), "--dataset", str(DEV), "--vocab", vocab,
     "--out", str(RUNS/"dataset-stats-v22.md")], "stats")

# ---------------- 10. upload results to HF ----------------
log("uploading results")
api = HfApi(token=TOK)
api.create_repo(RESULTS_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=RESULTS_REPO, repo_type="dataset", folder_path=str(RUNS),
                  path_in_repo="runs", commit_message="v2.2 benchmark results")
log("DONE — results at https://huggingface.co/datasets/" + RESULTS_REPO)
if skipped: log("skipped:", "; ".join(skipped))
