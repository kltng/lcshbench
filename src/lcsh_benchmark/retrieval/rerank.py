"""Rerank the top-N candidates of an L1 selection submission with a
cross-encoder, emitting a new selection submission (L2)."""
import argparse
import json

from .retrieve import build_query_text


def _rerank_query(record: dict) -> str:
    parts = []
    if record.get("title"):
        parts.append(record["title"])
    if record.get("title_vernacular"):
        parts.append(record["title_vernacular"])
    for a in record.get("authors", [])[:3]:
        parts.append(a)
    if record.get("abstract"):
        parts.append(record["abstract"][:600])
    return " ".join(parts) or build_query_text(record)


def run(records: list[dict], l1_submission: dict, reranker, top_n: int,
        system: str) -> dict:
    preds_in = l1_submission["predictions"]
    preds_out: dict[str, list[str]] = {}
    for r in records:
        ranked = preds_in.get(r["id"], [])
        head, tail = ranked[:top_n], ranked[top_n:]
        reordered = reranker.rerank(_rerank_query(r), head) if head else []
        preds_out[r["id"]] = reordered + tail
    return {"system": system, "task": "selection", "predictions": preds_out}


def main() -> None:
    from .rerank_backends import FakeReranker, LocalCrossEncoderBackend
    ap = argparse.ArgumentParser(description="Rerank an L1 submission (L2).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--l1-submission", required=True)
    ap.add_argument("--backend", required=True, choices=["fake", "local"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    records = json.load(open(a.dataset, encoding="utf-8"))
    l1 = json.load(open(a.l1_submission, encoding="utf-8"))
    rr = FakeReranker() if a.backend == "fake" else LocalCrossEncoderBackend(a.model)
    base = l1.get("system", "l1")
    sub = run(records, l1, rr, a.top_n, system=f"{base}+rerank-{rr.name}")
    json.dump(sub, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(sub['predictions'])} reranked predictions -> {a.out}")
