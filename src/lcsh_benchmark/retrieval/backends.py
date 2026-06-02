"""Embedding backends behind one interface.

EmbeddingBackend.encode(texts) -> float32 (n, dim) L2-normalized array, so
retrieval is plain cosine = dot product. Local backends are free; the
OpenRouter backend meters tokens against a Ledger.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Protocol

import numpy as np


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class EmbeddingBackend(Protocol):
    name: str
    def encode(self, texts: list[str]) -> np.ndarray: ...
    def cost_usd(self, texts: list[str]) -> float: ...


class FakeBackend:
    """Deterministic hash-based vectors for tests (no model, no network)."""
    def __init__(self, dim: int = 16):
        self.dim = dim
        self.name = f"fake-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            buf = (h * ((self.dim // len(h)) + 1))[: self.dim]
            out[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return _l2_normalize(out)

    def cost_usd(self, texts: list[str]) -> float:
        return 0.0


class LocalSTBackend:
    """sentence-transformers model run locally (free). Requires the
    `local-embed` extra: uv sync --extra local-embed.

    truncate_dim applies Matryoshka truncation (then renormalization via
    normalize_embeddings) — set 256 to match the on-device lcsh.db. The dim is
    folded into .name so each (model, dim) gets its own index dir.

    device pins the torch device. Pass device="cpu" for the query-retrieval path:
    encoding a few thousand long query texts on Apple MPS reliably deadlocks the
    Metal shader cache (0% CPU hang), while the bulk index build on MPS is fine.
    The .name (hence the index dir) is device-independent, so a CPU-encoded query
    matches an MPS-built index."""
    def __init__(self, model: str, truncate_dim: int | None = None,
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, truncate_dim=truncate_dim, device=device)
        self.name = model.split("/")[-1]
        if truncate_dim:
            self.name = f"{self.name}-d{truncate_dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True,
                              convert_to_numpy=True, batch_size=128)
        return v.astype(np.float32)

    def cost_usd(self, texts: list[str]) -> float:
        return 0.0


# OpenRouter embedding prices (USD per 1M input tokens); verified at build time.
OPENROUTER_EMBED_PRICES = {
    "openai/text-embedding-3-small": 0.02,
    "openai/text-embedding-3-large": 0.13,
}


def _approx_tokens(text: str) -> int:
    # ~4 chars/token heuristic; only used for pre-charge cost estimation.
    return max(1, len(text) // 4)


class OpenRouterEmbedBackend:
    """OpenRouter /api/v1/embeddings (OpenAI-compatible). Meters cost."""
    URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(self, model: str, ledger=None, api_key: str | None = None):
        import httpx
        self.model = model
        self.name = model.split("/")[-1]
        self.ledger = ledger
        self.api_key = api_key or os.environ["openrouter_api_key"]
        self._client = httpx.Client(timeout=120.0)
        self._price = OPENROUTER_EMBED_PRICES.get(model, 0.13)

    def cost_usd(self, texts: list[str]) -> float:
        toks = sum(_approx_tokens(t) for t in texts)
        return self._price * toks / 1_000_000

    def _post_with_retry(self, texts: list[str], retries: int = 6) -> dict:
        """POST with backoff. Retries transient HTTP errors AND malformed/
        error 200s (non-JSON bodies, missing 'data') — the latter is what
        crashed an unguarded run. Surfaces the body snippet on final failure."""
        last = "unknown"
        for attempt in range(retries):
            try:
                r = self._client.post(
                    self.URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": texts},
                )
                if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                    last = f"status {r.status_code}: {r.text[:200]}"
                    raise RuntimeError(last)
                r.raise_for_status()
                try:
                    j = r.json()
                except Exception:
                    last = f"non-JSON body (status {r.status_code}): {r.text[:200]!r}"
                    raise RuntimeError(last)
                if "data" not in j:
                    last = f"no 'data' in response: {str(j)[:200]}"
                    raise RuntimeError(last)
                return j
            except Exception as e:
                last = str(e) or repr(e)
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"OpenRouter embeddings failed after {retries} tries: {last}")
                time.sleep(min(2 ** attempt, 30))  # 1,2,4,8,16,30s cap
        raise RuntimeError("unreachable")

    def encode(self, texts: list[str]) -> np.ndarray:
        cost = self.cost_usd(texts)
        if self.ledger is not None:
            self.ledger.charge(cost, {"backend": self.name, "n": len(texts),
                                      "kind": "embed"})
        data = self._post_with_retry(texts)["data"]
        # OpenAI/OpenRouter may return embeddings out of input order; sort by index.
        data = sorted(data, key=lambda d: d.get("index", 0))
        m = np.array([d["embedding"] for d in data], dtype=np.float32)
        return _l2_normalize(m)
