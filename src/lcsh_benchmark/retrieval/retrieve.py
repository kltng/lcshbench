"""Embed queries with a backend and retrieve top-k vocab labels per record,
emitting a standard selection submission."""
import argparse
import json

import numpy as np

from .embed_index import Index, build


def build_query_text(record: dict) -> str:
    parts = []
    if record.get("title"):
        parts.append(record["title"])
    if record.get("title_vernacular"):
        parts.append(record["title_vernacular"])
    for a in record.get("authors", []):
        parts.append(a)
    for a in record.get("authors_vernacular", []) or []:
        parts.append(a)
    if record.get("abstract"):
        parts.append(record["abstract"])
    if record.get("toc"):
        parts.append(record["toc"][:2000])
    return "\n\n".join(parts)


def top_k(index: Index, query_vec: np.ndarray, k: int) -> list[str]:
    if index.vectors.shape[0] == 0:
        return []
    scores = index.vectors @ query_vec.astype(np.float32)
    n = min(k, scores.shape[0])
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [index.labels[i] for i in top]


def _encode_batched(backend, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode in batches so API backends stay under per-request token/input
    limits. Local/fake backends are unaffected (one batch if batch_size large)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    chunks = [backend.encode(texts[i:i + batch_size])
              for i in range(0, len(texts), batch_size)]
    return np.vstack(chunks)


def run(records: list[dict], index: Index, backend, k: int, system: str,
        query_batch_size: int = 128) -> dict:
    queries = [build_query_text(r) for r in records]
    qvecs = _encode_batched(backend, queries, query_batch_size)
    preds = {}
    for r, qv in zip(records, qvecs):
        preds[r["id"]] = top_k(index, qv, k)
    return {"system": system, "task": "selection", "predictions": preds}


def main() -> None:
    from .backends import FakeBackend, LocalSTBackend, OpenRouterEmbedBackend
    from ..ledger import Ledger
    ap = argparse.ArgumentParser(description="Retrieve top-k per record.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--vocab", default="data/vocab/vocab.jsonl")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--backend", required=True, choices=["fake", "local", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--truncate-dim", type=int, default=None,
                    help="Matryoshka truncation for local backend (e.g. 256); "
                         "must match the index it retrieves against")
    ap.add_argument("--device", default=None,
                    help="torch device for local backend; use 'cpu' to avoid the "
                         "Apple MPS query-encode deadlock (see LocalSTBackend)")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default="data/index/ledger.json")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="index-build embed batch (lower to avoid GPU OOM, e.g. 32)")
    a = ap.parse_args()

    ledger = Ledger(a.ledger, a.max_cost)
    if a.backend == "fake":
        be = FakeBackend()
    elif a.backend == "local":
        be = LocalSTBackend(a.model, truncate_dim=a.truncate_dim, device=a.device)
    else:
        be = OpenRouterEmbedBackend(a.model, ledger=ledger)

    index = build(a.vocab, be, a.index_dir, batch_size=a.batch_size)
    records = json.load(open(a.dataset, encoding="utf-8"))
    if a.limit:
        records = records[: a.limit]
    sub = run(records, index, be, a.k, system=f"retrieval-{be.name}")
    json.dump(sub, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {len(sub['predictions'])} predictions -> {a.out}")
