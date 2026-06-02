# src/lcsh_benchmark/baselines/frequency.py
"""Frequency-floor baseline: predict the globally most-frequent headings.

A standard extreme-multilabel lower bound — the same ranked list for every
record, ordered by descending corpus frequency (ties broken alphabetically).
"""
import argparse
import json
from collections import Counter


def ranked_by_frequency(records: list[dict]) -> list[str]:
    counts: Counter[str] = Counter()
    for r in records:
        for h in r["ground_truth_lcsh_merged"]:
            counts[h] += 1
    return [h for h, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def frequency_submission(records: list[dict], k: int,
                         rank_from: list[dict] | None = None) -> dict:
    """Top-k most-frequent headings as the prediction for every record.

    Frequencies are counted over `rank_from` when given (e.g. dev), so the
    held-out test set — which carries no GT — can be predicted for by ranking
    from dev. Otherwise frequencies come from `records` themselves.
    """
    ranked = ranked_by_frequency(rank_from if rank_from is not None else records)[:k]
    return {
        "system": f"frequency-floor-top{k}",
        "task": "selection",
        "predictions": {r["id"]: list(ranked) for r in records},
    }


def run(dataset: str, out: str, k: int = 200, freq_from: str | None = None) -> dict:
    """Emit a frequency-floor submission JSON for `dataset` to `out`."""
    records = json.load(open(dataset, encoding="utf-8"))
    rank_from = json.load(open(freq_from, encoding="utf-8")) if freq_from else None
    sub = frequency_submission(records, k, rank_from=rank_from)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False, indent=1)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description="Frequency-floor baseline submission.")
    ap.add_argument("--dataset", required=True, help="records to predict for")
    ap.add_argument("--out", required=True, help="write the submission JSON here")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--freq-from", default=None,
                    help="rank from this dataset (e.g. dev) — needed for the GT-less test set")
    a = ap.parse_args()
    sub = run(a.dataset, a.out, a.k, a.freq_from)
    print({"system": sub["system"], "records": len(sub["predictions"]), "k": a.k})
