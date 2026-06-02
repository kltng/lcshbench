# src/lcsh_benchmark/scaleup/build_v2.py
"""Build v2 dev/test datasets from the selection manifest + enriched corpus."""
import argparse
import json
from pathlib import Path

from ..build import hash_gt, stratified_split
from .assemble import assemble_record
from .match import match_key

_GT_KEYS = ("ground_truth_lcsh", "ground_truth_lcsh_merged",
            "ground_truth_lcsh_unanimous", "heading_types")


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    return {r["key"]: r for r in (m["core"] + m["breadth"])}


def gather_corpus(corpus_dir: str, selected: set) -> dict:
    """match_key -> {source: corpus_record} for selected keys (streamed)."""
    per_key: dict[str, dict[str, dict]] = {}
    for shard in sorted(Path(corpus_dir).rglob("*.jsonl")):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                k = match_key(r)
                if k in selected:
                    per_key.setdefault(k, {})[r["source"]] = r
    return per_key


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def build(manifest_path: str, corpus_dir: str, out_dir: str,
          test_frac: float = 0.15, seed: int = 13) -> dict:
    manifest = load_manifest(manifest_path)
    per_key = gather_corpus(corpus_dir, set(manifest))

    records, corpus_miss, empty_gt = [], 0, 0
    for key, mrec in manifest.items():
        ps = per_key.get(key)
        if not ps:
            corpus_miss += 1
            continue
        rec = assemble_record(key, mrec, ps)
        if rec is None:
            empty_gt += 1
            continue
        records.append(rec)

    dev, test = stratified_split(records, test_frac, seed)
    out = Path(out_dir)
    (out / "dev").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    _write(out / "dev/dataset_dev.json", dev)

    hashed, public = {}, []
    for r in test:
        hashed[r["id"]] = hash_gt(r["ground_truth_lcsh_merged"])
        public.append({k: v for k, v in r.items() if k not in _GT_KEYS})
    _write(out / "test/dataset_test.json", public)
    _write(out / "test/gt_test.hashed.json", hashed)
    return {"records": len(records), "dev": len(dev), "test": len(test),
            "dropped": corpus_miss + empty_gt, "corpus_miss": corpus_miss, "empty_gt": empty_gt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/v2_manifest.json")
    ap.add_argument("--corpus", default="data/raw/v2/corpus")
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    print(build(a.manifest, a.corpus, a.out, a.test_frac, a.seed))
