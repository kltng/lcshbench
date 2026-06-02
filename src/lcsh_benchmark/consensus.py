# src/lcsh_benchmark/consensus.py
"""Merge per-catalog LCSH assignments into a consensus set.

v1 default: union across catalogs, deduped on normalized form. Inclusion
threshold (catalog_count >= 2) is applied by the build pipeline, not here.
"""
from .normalize import normalize_label


def catalog_count(gt_per_catalog: dict[str, list[str]]) -> int:
    return sum(1 for headings in gt_per_catalog.values() if headings)


def merge_union(gt_per_catalog: dict[str, list[str]]) -> list[str]:
    seen: dict[str, str] = {}  # normalized -> first surface form
    for headings in gt_per_catalog.values():
        for h in headings:
            key = normalize_label(h)
            if key and key not in seen:
                seen[key] = h
    return sorted(seen.values(), key=str.lower)
