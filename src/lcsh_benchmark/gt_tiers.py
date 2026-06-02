"""Consensus tiers of a record's GT, from the per-catalog assignments.

Makes the union/majority/unanimous distinction explicit so 'consensus' is not
overstated: the released `ground_truth_lcsh_merged` is the UNION (includes
single-source headings); these tiers expose the heading-level agreement behind it.

NOTE — tiers OVERLAP: ``majority`` (≥2 sources) is a strict superset of
``unanimous`` (all sources); ``single`` (exactly 1 source) is disjoint from both.
Summing ``unanimous + majority + single`` double-counts unanimous headings and
does NOT yield a partition of the GT. Use ``majority + single`` for a true
partition (all unique headings, no double-counting)."""
from collections import Counter


def heading_tiers(record: dict) -> dict:
    per = record.get("ground_truth_lcsh", {})
    n_sources = len(per)
    counts = Counter(h for hs in per.values() for h in set(hs))
    unanimous = [h for h, c in counts.items() if c == n_sources and n_sources > 0]
    majority = [h for h, c in counts.items() if c >= 2]
    single = [h for h, c in counts.items() if c == 1]
    return {"unanimous": sorted(unanimous), "majority": sorted(majority),
            "single": sorted(single)}


def corpus_tier_counts(records: list[dict]) -> dict:
    out = {"unanimous": 0, "majority": 0, "single": 0}
    for r in records:
        t = heading_tiers(r)
        for k in out:
            out[k] += len(t[k])
    return out
