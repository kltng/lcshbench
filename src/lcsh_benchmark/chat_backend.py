"""Chat backends (OpenAI-compatible) with disk cache + ledger metering.

OpenRouterChat and DeepSeekNativeChat share one request shape and one Ledger.
Cost is pre-estimated (prompt chars + max_tokens) to enforce the cap BEFORE
calling, then trued-up from the response usage. Responses are cached by
(model, system, user, max_tokens) hash so reruns/resumes are free.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol


def run_concurrent(items: list, fn, workers: int = 8, label: str = "",
                   progress_every: int = 50) -> dict:
    """Map fn over items concurrently; fn(item) -> (key, value). Returns
    {key: value}. Used for per-record LLM calls (I/O-bound). The ledger is
    thread-safe and the response cache is per-file, so workers are safe."""
    results: dict = {}
    n = len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for fut in as_completed(futs):
            k, v = fut.result()
            results[k] = v
            done += 1
            if progress_every and (done % progress_every == 0 or done == n):
                print(f"[{label}] {done}/{n}", file=sys.stderr)
    return results


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_heading_list(text: str) -> list[str]:
    """Extract a heading list from an LLM reply: JSON array if present, else
    bullet/line list. Dedups preserving first-seen order."""
    items: list[str] = []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            items = [str(x).strip() for x in json.loads(m.group(0))]
        except Exception:
            items = []
    if not items:
        for line in text.splitlines():
            s = line.strip().lstrip("-*0123456789. ").strip()
            if s and not s.lower().startswith(("here", "the ", "these")):
                items.append(s)
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it); out.append(it)
    return out


class ChatBackend(Protocol):
    name: str
    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class _CachingMeteringChat:
    """Shared cache + cap-check + metering; subclasses implement _call()."""
    name: str
    price_in: float   # USD per 1M input tokens
    price_out: float  # USD per 1M output tokens

    def __init__(self, ledger=None, cache_dir: str = "data/chat_cache"):
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.calls = 0

    def _cache_key(self, system, user, max_tokens) -> Path:
        h = hashlib.sha256(
            f"{self.name}\x00{system}\x00{user}\x00{max_tokens}".encode()
        ).hexdigest()
        return self.cache_dir / self.name / f"{h}.json"

    def _estimate(self, system, user, max_tokens) -> float:
        tin = _approx_tokens(system) + _approx_tokens(user)
        return (self.price_in * tin + self.price_out * max_tokens) / 1_000_000

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        cp = self._cache_key(system, user, max_tokens)
        if cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))["content"]
        if self.ledger is not None:
            est = self._estimate(system, user, max_tokens)
            if self.ledger.would_exceed(est):
                from .ledger import BudgetExceeded
                raise BudgetExceeded(f"{self.name}: est ${est:.4f} exceeds cap")
        content, usage = self._call(system, user, max_tokens)
        self.calls += 1
        if self.ledger is not None:
            cost = (self.price_in * usage["prompt_tokens"]
                    + self.price_out * usage["completion_tokens"]) / 1_000_000
            self.ledger.charge(cost, {"backend": self.name, "kind": "chat"})
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"content": content, "usage": usage},
                                 ensure_ascii=False), encoding="utf-8")
        return content

    def _call(self, system, user, max_tokens) -> tuple[str, dict]:
        raise NotImplementedError


class FakeChat(_CachingMeteringChat):
    """Canned reply, no network — for tests."""
    def __init__(self, reply: str, price_in=0.0, price_out=0.0, ledger=None,
                 cache_dir="data/chat_cache", name="fake-chat"):
        super().__init__(ledger, cache_dir)
        self.name = name
        self.price_in, self.price_out = price_in, price_out
        self._reply = reply

    def _call(self, system, user, max_tokens):
        return self._reply, {
            "prompt_tokens": _approx_tokens(system) + _approx_tokens(user),
            "completion_tokens": _approx_tokens(self._reply)}


# Per-1M-token prices (USD), verified at run time. DeepSeek effective rates.
PRICES = {
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.14, 0.28),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "openai/gpt-4o-mini": (0.15, 0.60),
}


class _OpenAICompatChat(_CachingMeteringChat):
    base_url: str
    api_key_env: str

    def __init__(self, model, ledger=None, cache_dir="data/chat_cache"):
        import httpx
        super().__init__(ledger, cache_dir)
        self.model = model
        self.name = model.split("/")[-1]
        self.price_in, self.price_out = PRICES.get(model, (1.0, 3.0))
        self._key = os.environ[self.api_key_env]
        self._client = httpx.Client(timeout=180.0)

    def _call(self, system, user, max_tokens):
        last = "?"
        for attempt in range(6):
            try:
                r = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={"model": self.model, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]},
                )
                if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                    last = f"status {r.status_code}: {r.text[:200]}"
                    raise RuntimeError(last)
                r.raise_for_status()
                j = r.json()
                content = j["choices"][0]["message"]["content"]
                usage = j.get("usage", {
                    "prompt_tokens": _approx_tokens(system + user),
                    "completion_tokens": _approx_tokens(content)})
                return content, {"prompt_tokens": usage["prompt_tokens"],
                                 "completion_tokens": usage["completion_tokens"]}
            except Exception as e:
                last = str(e) or repr(e)
                if attempt == 5:
                    raise RuntimeError(f"{self.name} chat failed after 6 tries: {last}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("unreachable")


class OpenRouterChat(_OpenAICompatChat):
    base_url = "https://openrouter.ai/api/v1"
    api_key_env = "openrouter_api_key"


class DeepSeekNativeChat(_OpenAICompatChat):
    base_url = "https://api.deepseek.com"
    api_key_env = "deepseek_api_key"
