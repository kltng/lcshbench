"""Audit likely non-LCSH leakage in committed GT (H2 disclosure).

Flags topical/geographic GT headings unreachable from the vocab even at root
(candidate non-LCSH/foreign-thesaurus strings) and those with non-Latin scripts.
Quantifies the leakage the MARC source filter will remove at the v2.1
re-extract; modifies no data."""
import argparse
import json

from .metrics import root_key
from .retrieval.score_retrieval import VocabKeys, load_vocab_keys, DEFAULT_VOCAB

_CHECK_TYPES = {"topical", "geographic"}


def _non_latin(s: str) -> bool:
    return any(ord(c) > 0x24F and c.isalpha() for c in s)


def audit(records, vocab: VocabKeys) -> dict:
    total = root_unreachable = non_latin = 0
    examples = []
    for r in records:
        for h in r["ground_truth_lcsh_merged"]:
            if r["heading_types"].get(h, {}).get("type") not in _CHECK_TYPES:
                continue
            total += 1
            unreached = root_key(h) not in vocab.root
            nl = _non_latin(h)
            if unreached:
                root_unreachable += 1
            if nl:
                non_latin += 1
            if (unreached or nl) and len(examples) < 50:
                examples.append(h)
    return {"total": total, "root_unreachable": root_unreachable,
            "non_latin": non_latin,
            "root_unreachable_frac": root_unreachable / total if total else 0.0,
            "non_latin_frac": non_latin / total if total else 0.0,
            "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit non-LCSH leakage in GT.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    a = ap.parse_args()
    with open(a.dataset, encoding="utf-8") as f:
        recs = json.load(f)
    rep = audit(recs, load_vocab_keys(a.vocab))
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=1))
    print("examples:", rep["examples"][:20])
