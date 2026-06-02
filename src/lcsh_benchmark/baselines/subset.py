# src/lcsh_benchmark/baselines/subset.py
"""Language-stratified subset of a dataset, for cost-bounded baseline runs.

The neural lcsh-onnx baseline is ~0.13 min/record; on the full 19K-record v2
dev set that is ~43h. A language-stratified subset gives a representative
leaderboard (preserving the per-language breakdown, incl. the Russian outlier)
at a fraction of the cost. Deterministic given --seed, so it is reproducible
and the same subset can be scored across systems.
"""
import argparse
import json
import random
from collections import defaultdict


def stratified_subset(records: list[dict], target: int, seed: int = 13,
                      floor: int = 5) -> list[dict]:
    """Proportional language-stratified sample of ~`target` records.

    Each language gets `round(target * share)` records but at least `floor`
    (so small languages still appear for per-language metrics), capped at the
    language's available count. Order is preserved from the input after a
    seeded per-language shuffle, so re-running yields the identical subset.
    """
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lang[r["language_code"]].append(r)

    n = len(records)
    rng = random.Random(seed)
    picked: list[dict] = []
    for lang in sorted(by_lang):
        pool = by_lang[lang]
        want = max(floor, round(target * len(pool) / n))
        want = min(want, len(pool))
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        picked.extend(pool[i] for i in sorted(idx[:want]))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Language-stratified dataset subset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--floor", type=int, default=5)
    a = ap.parse_args()
    records = json.load(open(a.dataset, encoding="utf-8"))
    sub = stratified_subset(records, a.target, a.seed, a.floor)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    from collections import Counter
    dist = Counter(r["language_code"] for r in sub)
    print({"n": len(sub), "languages": dict(sorted(dist.items(), key=lambda kv: -kv[1]))})
