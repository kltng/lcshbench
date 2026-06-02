"""Cross-encoder rerank backends behind one interface.

RerankBackend.rerank(query, candidates) -> candidates reordered best-first.
FakeReranker is for tests (no model); LocalCrossEncoderBackend wraps a
sentence-transformers CrossEncoder (jina, bge-reranker). Local = free.
"""
from __future__ import annotations

from typing import Protocol


class RerankBackend(Protocol):
    name: str
    def rerank(self, query: str, candidates: list[str]) -> list[str]: ...


def _tokens(s: str) -> set[str]:
    return {t for t in s.lower().replace("--", " ").split() if t}


class FakeReranker:
    """Deterministic lexical-overlap reranker for tests (no model/network)."""
    name = "fake-reranker"

    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        q = _tokens(query)
        scored = sorted(candidates,
                        key=lambda c: (-len(q & _tokens(c)), candidates.index(c)))
        return scored


class LocalCrossEncoderBackend:
    """sentence-transformers CrossEncoder run locally (free). Requires the
    `local-embed` extra. Models: jinaai/jina-reranker-v2-base-multilingual,
    BAAI/bge-reranker-v2-m3."""
    def __init__(self, model: str, max_pairs: int = 64):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, trust_remote_code=True)
        self.name = model.split("/")[-1]

    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [candidates[i] for i in order]
