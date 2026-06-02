#!/usr/bin/env python3
"""Build leak-free (anchor, positive) training pairs for the embedder fine-tune.

The original data/finetune/train_pairs.jsonl was drawn from the FULL dev set,
which includes the dev-2K leaderboard subset — so the fine-tune had seen ~72% of
the eval records (train-on-test). This rebuilds the pairs from `dev MINUS dev-2K`,
excluding every eval record by id AND by (title, first-author), so the dev-2K
evaluation is clean.

    uv run --extra local-embed scripts/build_finetune_pairs.py

- anchor   = title + vernacular + authors + abstract (space-joined, capped at
             2000 chars) — the original recipe, reproduced exactly so the ONLY
             change from the leaky run is removing the eval records. (Appending
             toc, as an earlier attempt did, produced 7k-char anchors that OOM'd
             the A10G at batch 32; the original capped near 2000.)
- positive = each merged-GT heading that is exact-reachable in the vocab
             (normalize_label(h) is a vocab label) — exactly the scorer's target.

Writes data/finetune/train_pairs.jsonl and asserts zero dev-2K overlap.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from lcsh_benchmark.normalize import normalize_label
from lcsh_benchmark.retrieval.score_retrieval import load_vocab_keys

DEV = "data/v2/dev/dataset_dev.json"
DEV2K = "data/v2/dev/dataset_dev_subset2k.json"
OUT = Path("data/finetune/train_pairs.jsonl")
MAX_ANCHOR_CHARS = 2000


def build_anchor(r: dict) -> str:
    authors = "; ".join(a for a in (r.get("authors") or []) if a)
    parts = [r.get("title"), r.get("title_vernacular"), authors, r.get("abstract")]
    anchor = " ".join(p.strip() for p in parts if p and p.strip())
    return anchor[:MAX_ANCHOR_CHARS]


def _key(r: dict) -> tuple[str, str]:
    t = re.sub(r"\s+", " ", (r.get("title") or "").strip().lower())
    au = r.get("authors") or []
    a0 = re.sub(r"\s+", " ", (au[0] if au else "").strip().lower())
    return (t, a0)


def main() -> None:
    dev = json.load(open(DEV, encoding="utf-8"))
    dev2k = json.load(open(DEV2K, encoding="utf-8"))
    vocab = load_vocab_keys()

    excl_ids = {r.get("id") for r in dev2k}
    excl_keys = {_key(r) for r in dev2k}
    print(f"full dev={len(dev)}  dev-2K excluded={len(dev2k)} "
          f"(ids={len(excl_ids)}, title+author keys={len(excl_keys)})")

    pairs, kept_recs, skipped_leak = [], 0, 0
    for r in dev:
        if r.get("id") in excl_ids or _key(r) in excl_keys:
            skipped_leak += 1
            continue
        anchor = build_anchor(r)
        if not anchor:
            continue
        positives = [h for h in (r.get("ground_truth_lcsh_merged") or [])
                     if normalize_label(h) in vocab.exact]
        if not positives:
            continue
        kept_recs += 1
        for h in positives:
            pairs.append({"anchor": anchor, "positive": h})

    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # verify zero leakage: no clean anchor equals a dev-2K record's anchor
    anchors = {p["anchor"] for p in pairs}
    dev2k_anchor = {build_anchor(r) for r in dev2k}
    overlap = len(anchors & dev2k_anchor)
    print(f"kept records={kept_recs}  pairs={len(pairs)}  unique anchors={len(anchors)}")
    print(f"records skipped as dev-2K leak={skipped_leak}")
    print(f"anchor==dev2k-anchor overlap: {overlap} (must be 0)")
    assert overlap == 0, "LEAK: clean anchors still overlap dev-2K"
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
