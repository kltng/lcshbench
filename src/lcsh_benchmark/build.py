# src/lcsh_benchmark/build.py
"""Per-record cleaning/typing pipeline + the dataset build CLI."""
from collections.abc import Callable

from .clean import clean_headings
from .consensus import catalog_count, merge_union
from .typing import type_heading


def process_record(rec: dict, lookup: Callable[[str], str | None]) -> tuple[dict, list[str]]:
    cleaned_gt: dict[str, list[str]] = {}
    all_dropped: list[str] = []
    for catalog, headings in rec["ground_truth_lcsh"].items():
        kept, dropped = clean_headings(headings)
        cleaned_gt[catalog] = kept
        all_dropped.extend(dropped)

    merged = merge_union(cleaned_gt)
    out = {
        **rec,
        "ground_truth_lcsh": cleaned_gt,
        "catalogs": sorted(k for k, v in cleaned_gt.items() if v),
        "catalog_count": catalog_count(cleaned_gt),
        "ground_truth_lcsh_merged": merged,
        "heading_types": {h: type_heading(h, lookup) for h in merged},
    }
    return out, all_dropped


import argparse
import hashlib
import json
import random
import sqlite3
from pathlib import Path

from .load import load_records, to_canonical
from .normalize import normalize_label
from .typing import make_db_lookup


def stratified_split(records: list[dict], test_frac: float, seed: int):
    by_lang: dict[str, list[dict]] = {}
    for r in records:
        by_lang.setdefault(r["language_code"], []).append(r)
    dev, test = [], []
    rng = random.Random(seed)
    for lang in sorted(by_lang):
        group = sorted(by_lang[lang], key=lambda r: r["id"])
        rng.shuffle(group)
        n_test = round(len(group) * test_frac)
        test.extend(group[:n_test])
        dev.extend(group[n_test:])
    dev.sort(key=lambda r: r["id"])
    test.sort(key=lambda r: r["id"])
    return dev, test


def hash_gt(headings: list[str]) -> list[str]:
    return [hashlib.sha256(normalize_label(h).encode()).hexdigest() for h in headings]


SEEDS = [
    ("../lcsh-database-validation/data/dataset_100_eng.json", "eng100"),
    ("../lcsh-eval/data/dataset.json", "multi500"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="../lcsh-onnx/db-builder/dist/lcsh.db")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-catalogs", type=int, default=2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    lookup = make_db_lookup(conn)

    canonical = []
    for path, source in SEEDS:
        for rec in load_records(path):
            canonical.append(to_canonical(rec, source))

    processed, dropped_log = [], []
    for rec in canonical:
        out, dropped = process_record(rec, lookup)
        if out["catalog_count"] >= args.min_catalogs:
            processed.append(out)
            for d in dropped:
                dropped_log.append((out["id"], d))

    dev, test = stratified_split(processed, args.test_frac, args.seed)
    out_dir = Path(args.out)
    (out_dir / "dev").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    _write(out_dir / "dev/dataset_dev.json", dev)

    test_public = []
    hashed = {}
    for r in test:
        hashed[r["id"]] = hash_gt(r["ground_truth_lcsh_merged"])
        pub = {k: v for k, v in r.items()
               if k not in ("ground_truth_lcsh", "ground_truth_lcsh_merged", "heading_types")}
        test_public.append(pub)
    _write(out_dir / "test/dataset_test.json", test_public)
    _write(out_dir / "test/gt_test.hashed.json", hashed)

    _write_log(Path("docs/cleaning_log.md"), processed, dev, test, dropped_log)
    print(f"dev={len(dev)} test={len(test)} dropped_artifacts={len(dropped_log)}")
    conn.close()


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _write_log(path: Path, processed, dev, test, dropped_log) -> None:
    from collections import Counter
    langs = Counter(r["language_code"] for r in processed)
    auth = Counter(t["authority"] for r in processed for t in r["heading_types"].values())
    lines = [
        "# Phase 1 cleaning log", "",
        f"- Records after >=2-catalog filter: {len(processed)} (dev {len(dev)}, test {len(test)})",
        f"- MARC artifacts dropped: {len(dropped_log)}", "",
        "## Per-language record counts", "",
        *[f"- {k}: {v}" for k, v in sorted(langs.items())], "",
        "## Heading authority distribution (merged GT)", "",
        *[f"- {k}: {v}" for k, v in auth.most_common()], "",
        "## Dropped MARC artifacts", "",
        *[f"- `{rid}`: `{h}`" for rid, h in dropped_log],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
