# src/lcsh_benchmark/baselines/build_distill_cache.py
"""Prebuild the lcsh-onnx route-A distilled-query cache for a benchmark dataset.

The lcsh-onnx pipeline distills each record's metadata into a concise ENGLISH
query (topical concepts + entities) via a local Gemma, and embeds THAT instead
of the raw bibliographic text (the "replace" strategy) — which ~doubles
multilingual retrieval recall. This is the slow, LLM step; it is cached and
resumable so the retrieval adapter (lcsh_onnx_adapter.py --strategy replace
--distill-cache <out>) can then run cheaply.

Run under the lcsh-onnx db-builder venv (provides onnxruntime + the pipeline):

    cd ../lcsh-onnx/db-builder
    uv run python <abs>/src/lcsh_benchmark/baselines/build_distill_cache.py \
        --dataset <abs>/data/v2/dev/dataset_dev_subset2k.json \
        --out <abs>/data/distill/distill_v2_dev_subset2k.json [--limit N]

~5-6s/record on a local Gemma-4-E2B (ONNX); saves every 10, so it resumes.
"""
import argparse
import json
import sys
from pathlib import Path

from lcsh_db_builder.distill_cli import build_cache_for_records
from lcsh_db_builder.lib.selectors import make_selector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    help="benchmark dataset JSON (a list of records, or {records:[...]})")
    ap.add_argument("--out", required=True, help="distill cache JSON (id -> query)")
    ap.add_argument("--selector-backend", default="onnx")
    ap.add_argument("--selector-model", default="onnx-community/gemma-4-E2B-it-ONNX")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        ds = json.load(f)
    records = ds["records"] if isinstance(ds, dict) and "records" in ds else ds
    if args.limit:
        records = records[:args.limit]

    selector = make_selector(args.selector_backend, args.selector_model)
    cache = build_cache_for_records(records, selector.generate, Path(args.out))
    print(f"[build-distill-cache] cached {len(cache)} distilled queries -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
