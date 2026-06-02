"""Task-B (retrieval) scoring against true vocabulary reachability.

A GT heading is *exact-reachable* if its normalized form is a vocab label, and
*root-reachable* if its base (subdivisions stripped) is. Exact metrics are scored
over exact-reachable GT, root metrics over root-reachable GT, so neither is
penalized for labels that cannot be retrieved from the released vocabulary.
"""
import argparse
import json
from dataclasses import dataclass

from ..metrics import norm_key, root_key
from ..score import KS, load_submission, render_report, score_selection

DEFAULT_VOCAB = "data/vocab/vocab.jsonl"


@dataclass
class VocabKeys:
    exact: set
    root: set


def load_vocab_keys(path: str = DEFAULT_VOCAB) -> VocabKeys:
    exact, root = set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            label = json.loads(line)["label"]
            exact.add(norm_key(label))
            root.add(root_key(label))
    return VocabKeys(exact, root)


def reachable(records, vocab: VocabKeys, mode: str):
    target = vocab.exact if mode == "exact" else vocab.root
    keyfn = norm_key if mode == "exact" else root_key
    out, kept = [], 0
    for r in records:
        keep = [h for h in r["ground_truth_lcsh_merged"] if keyfn(h) in target]
        nr = dict(r)
        nr["ground_truth_lcsh_merged"] = keep
        out.append(nr)
        kept += len(keep)
    return out, kept


def score(records, predictions, ks=KS, vocab: VocabKeys | None = None,
          vocab_path: str = DEFAULT_VOCAB) -> dict:
    vocab = vocab or load_vocab_keys(vocab_path)
    results = {"total_gt": sum(len(r["ground_truth_lcsh_merged"]) for r in records)}
    for mode in ("exact", "root"):
        recs, kept = reachable(records, vocab, mode)
        m = score_selection(recs, predictions, mode, ks)
        m["reachable_gt"] = kept
        results[mode] = m
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a retrieval (Task B) submission.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    records = json.load(open(a.dataset, encoding="utf-8"))
    sub = load_submission(a.submission)
    results = score(records, sub["predictions"], vocab_path=a.vocab)
    print(render_report("selection", results))
    print(f"total GT: {results['total_gt']}  "
          f"exact-reachable: {results['exact']['reachable_gt']}  "
          f"root-reachable: {results['root']['reachable_gt']}")
    if a.out:
        from ..ci import metric_cis
        cis = metric_cis(records, sub["predictions"], "selection", vocab_path=a.vocab)
        for mode in ("exact", "root"):
            results[mode]["cis"] = {m: list(cis[m][mode]) for m in cis}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"system": sub.get("system", "?"), "task": "selection",
                       "results": results}, f, ensure_ascii=False, indent=1)
