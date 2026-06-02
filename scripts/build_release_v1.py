"""Build the LCSHBench v1.0 dataset release (PRIVATE staging) on HF.

Release = the v2.2 build, published as v1.0. Configs:
  dev / test / dev_2k  (benchmark splits, JSONL — native schema, Viewer-friendly)
  consensus_matched    (1.29M-row provenance/concordance Parquet, hashed IDs)
  vocab                (retrieval vocabulary)
NOT released: the fine-tuned model, te3 indices (OpenAI ToS), raw MARC.
"""
import json, os, shutil, hashlib
from pathlib import Path
from huggingface_hub import HfApi

def _anon(r: dict) -> dict:
    """Hash the match-key id (same scheme as consensus_matched, so the same work
    links across configs) and drop raw OCLC/LCCN."""
    r = dict(r)
    if r.get("id"):
        r["id"] = hashlib.sha256(str(r["id"]).encode()).hexdigest()[:16]
    r.pop("oclc", None); r.pop("lccn", None)
    return r

REPO = "kltng/lcsh-benchmark"
STAGE = Path("data/release/v1.0");
SRC = Path("data/v2.2")
TOK = os.environ["HF_TOKEN"]

def to_jsonl(src_json: Path, dst: Path):
    recs = json.load(open(src_json, encoding="utf-8"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(_anon(r), ensure_ascii=False) + "\n")
    return len(recs)

if STAGE.exists(): shutil.rmtree(STAGE)
n_dev = to_jsonl(SRC/"dev/dataset_dev.json", STAGE/"dev/dev.jsonl")
n_sub = to_jsonl(SRC/"dev/dataset_dev_subset2k.json", STAGE/"dev_2k/dev_2k.jsonl")
n_test = to_jsonl(SRC/"test/dataset_test.json", STAGE/"test/test.jsonl")
# re-key the hashed test answers with the SAME id hash so scoring still joins
_gt = json.load(open(SRC/"test/gt_test.hashed.json", encoding="utf-8"))
_gt_anon = {hashlib.sha256(str(k).encode()).hexdigest()[:16]: v for k, v in _gt.items()}
json.dump(_gt_anon, open(STAGE/"test/gt_test.hashed.json", "w", encoding="utf-8"), ensure_ascii=False)
(STAGE/"consensus_matched").mkdir(parents=True, exist_ok=True)
shutil.copy("data/lcsh_consensus_matched.parquet", STAGE/"consensus_matched/consensus_matched.parquet")
(STAGE/"vocab").mkdir(parents=True, exist_ok=True)
shutil.copy("data/vocab/vocab.jsonl", STAGE/"vocab/vocab.jsonl")

CARD = f"""---
license: cc0-1.0
language: [en]
pretty_name: "LCSHBench v1.0"
task_categories: [text-classification, text-retrieval]
tags: [library-science, subject-indexing, LCSH, cataloging, multilingual]
configs:
  - config_name: dev
    data_files: dev/dev.jsonl
  - config_name: dev_2k
    data_files: dev_2k/dev_2k.jsonl
  - config_name: test
    data_files: test/test.jsonl
  - config_name: consensus_matched
    data_files: consensus_matched/consensus_matched.parquet
  - config_name: vocab
    data_files: vocab/vocab.jsonl
---

# LCSHBench v1.0

A multilingual, **consensus-grounded** benchmark for Library of Congress Subject
Heading (LCSH) assignment. Ground truth is the agreement of independent research
libraries (Columbia, Harvard, Princeton) cataloging the same work.

## Configs
| config | rows | description |
|---|---|---|
| `dev` | {n_dev:,} | development split (full per-catalog + merged + unanimous GT) |
| `dev_2k` | {n_sub:,} | language-balanced 2K evaluation subset |
| `test` | {n_test:,} | held-out test inputs; answers in `gt_test.hashed.json` (SHA-256) |
| `consensus_matched` | 1,286,609 | every work cataloged by >=2 libraries: per-library headings (hashed IDs) — the provenance/concordance population |
| `vocab` | 515,281 | LCSH+LCGFT retrieval vocabulary |

## Inter-cataloger concordance (why consensus)
Over 465,187 works cataloged by all three libraries: **93.3%** share a
concept-level heading, only **0.2%** share none (median concept Jaccard 0.86) —
yet just **39.4%** assign identical exact sets and **35.6%** of assertions are
single-source. Subject *identification* is objective; *expression* is subjective.
Human ceiling (a library vs the consensus of the other two): **86.9%** exact /
**93.0%** concept recall.

## Not included
The fine-tuned embedder and te3 embedding indices are **not** released
(model not published; te3 vectors are derived from a hosted API). Raw MARC is
redistributable from the source libraries; rebuild instructions in the repo.

## License & citation
Data: CC0-1.0 (derived from CC0 bulk MARC). Identifiers are hashed.
Built by the v2.2 extraction pipeline. Citation: TBD.
"""
(STAGE/"README.md").write_text(CARD, encoding="utf-8")

api = HfApi(token=TOK)
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(STAGE),
                  commit_message="LCSHBench v1.0 (private staging)")
print(f"staged PRIVATE -> https://huggingface.co/datasets/{REPO}")
print(f"  dev={n_dev} dev_2k={n_sub} test={n_test}")
