# src/lcsh_benchmark/scaleup/sample.py
"""Tiered, language + LC-class stratified sampling with a music cap.

Both tiers prefer records that already carry input text (has_input=True);
within each tier, ties are randomized reproducibly by seed. Deterministic
given a seed.
"""
import random
from collections import defaultdict


def monograph_only(records: list[dict]) -> list[dict]:
    """Selection-time material filter (A3): keep only records flagged as
    monographs (leader/06 in {a,t} AND leader/07 == m, propagated through the
    match). Records without the flag are excluded. The corpus keeps all
    material so the mix can still be reported; this gates what is sampled."""
    return [r for r in records if r.get("is_monograph")]


def _take_language(records: list[dict], n: int, music_cap: float,
                   rng: random.Random) -> list[dict]:
    if n <= 0 or not records:
        return []
    # Seeded shuffle then a STABLE sort putting has_input records first:
    # input-preference dominates, ties are randomized reproducibly by seed.
    pool = list(records)
    rng.shuffle(pool)
    pool.sort(key=lambda r: not r["has_input"])

    music = [r for r in pool if r["lc_class"] == "M"]
    other = [r for r in pool if r["lc_class"] != "M"]
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in other:
        by_cls[r["lc_class"]].append(r)   # buckets keep the input-first order
    classes = sorted(by_cls)

    max_music = int(n * music_cap)
    n_other = n - min(max_music, len(music))
    spread: list[dict] = []
    i = 0
    while any(by_cls.values()) and len(spread) < n_other:
        c = classes[i % len(classes)]
        if by_cls[c]:
            spread.append(by_cls[c].pop(0))   # pop front = highest-priority first
        i += 1

    chosen = spread + music[:min(max_music, n - len(spread))]
    if len(chosen) < n:                       # backfill from NON-music only (preserve cap)
        chosen_ids = {id(r) for r in chosen}
        rest = [r for r in pool if id(r) not in chosen_ids and r["lc_class"] != "M"]
        chosen += rest[:n - len(chosen)]
    return chosen[:n]


def stratified_sample(matched: list[dict], core_targets: dict[str, int],
                      breadth_targets: dict[str, int], music_cap: float,
                      seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in matched:
        by_lang[r["lang"]].append(r)

    used: set[str] = set()
    core: list[dict] = []
    for lang, n in core_targets.items():
        avail = [r for r in by_lang.get(lang, []) if r["key"] not in used]
        picked = _take_language(avail, n, music_cap, rng)
        used |= {r["key"] for r in picked}
        core += picked

    breadth: list[dict] = []
    for lang, n in breadth_targets.items():
        avail = [r for r in by_lang.get(lang, []) if r["key"] not in used]
        picked = _take_language(avail, n, music_cap, rng)
        used |= {r["key"] for r in picked}
        breadth += picked

    return core, breadth
