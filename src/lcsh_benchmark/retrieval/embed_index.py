"""Embed the vocab into a cached numpy index, one per (backend, vocab).

Index layout under <index_dir>/<backend-name>/: vectors.npy (float32, n×dim,
L2-normalized, row-aligned with) meta.json (labels + uris + authorities) +
progress.json (resumable-build checkpoint).

The build is **resumable and checkpointed**: vectors.npy is a memory-mapped
.npy filled batch-by-batch, with progress.json recording how many rows are
done. A failed batch (e.g. a flaky API response) loses at most that batch — a
re-run resumes from the checkpoint and never re-embeds completed rows (no
wasted spend). Memory-mapping also keeps peak RAM low for large-dim models.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


@dataclass
class Index:
    vectors: np.ndarray
    labels: list[str]
    uris: list[str]
    authorities: list[str]
    from_cache: bool = False


def _load_vocab(vocab_path: str):
    labels, uris, auths = [], [], []
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels.append(r["label"]); uris.append(r["uri"])
            auths.append(r["authority"])
    return labels, uris, auths


def _is_complete(vec_path: Path, prog_path: Path, n: int) -> bool:
    """True if a finished index already exists. Honors progress.json; falls
    back to a row-count check for legacy indexes built before checkpointing."""
    if not vec_path.exists():
        return False
    if prog_path.exists():
        # progress file is authoritative — a partial memmap already has the
        # full (n, dim) shape, so the row-count check below would be wrong here.
        return bool(json.loads(prog_path.read_text(encoding="utf-8")).get("complete"))
    try:  # legacy index built before checkpointing: trust a matching row count
        return np.load(vec_path, mmap_mode="r").shape[0] == n
    except Exception:
        return False


def build(vocab_path: str, backend, index_dir: str, batch_size: int = 256) -> Index:
    out = Path(index_dir) / backend.name
    vec_path = out / "vectors.npy"
    meta_path = out / "meta.json"
    prog_path = out / "progress.json"
    labels, uris, auths = _load_vocab(vocab_path)
    n = len(labels)

    if meta_path.exists() and _is_complete(vec_path, prog_path, n):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return Index(np.load(vec_path), meta["labels"], meta["uris"],
                     meta["authorities"], from_cache=True)

    out.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        meta_path.write_text(json.dumps(
            {"labels": labels, "uris": uris, "authorities": auths},
            ensure_ascii=False), encoding="utf-8")

    if n == 0:
        vectors = np.zeros((0, 0), dtype=np.float32)
        np.save(vec_path, vectors)
        prog_path.write_text(json.dumps({"done": 0, "dim": 0, "n": 0, "complete": True}))
        return Index(vectors, labels, uris, auths, from_cache=False)

    # Resume from a prior partial build if one exists.
    done, dim, mm = 0, None, None
    if prog_path.exists() and vec_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        done, dim = prog.get("done", 0), prog.get("dim")
        if dim is not None:
            mm = open_memmap(vec_path, mode="r+")

    i = done
    while i < n:
        block = backend.encode(labels[i:i + batch_size]).astype(np.float32)
        if mm is None:                       # first batch defines the dimension
            dim = int(block.shape[1])
            mm = open_memmap(vec_path, mode="w+", dtype=np.float32, shape=(n, dim))
        mm[i:i + block.shape[0]] = block
        mm.flush()
        i += int(block.shape[0])
        prog_path.write_text(json.dumps(
            {"done": i, "dim": dim, "n": n, "complete": i >= n}))

    del mm                                   # close the memmap
    return Index(np.load(vec_path), labels, uris, auths, from_cache=False)


def main() -> None:
    from .backends import FakeBackend, LocalSTBackend, OpenRouterEmbedBackend
    from ..ledger import Ledger
    ap = argparse.ArgumentParser(description="Build a vocab embedding index.")
    ap.add_argument("--vocab", default="data/vocab/vocab.jsonl")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--backend", required=True,
                    choices=["fake", "local", "openrouter"])
    ap.add_argument("--model", default=None, help="model id for local/openrouter")
    ap.add_argument("--truncate-dim", type=int, default=None,
                    help="Matryoshka truncation for local backend (e.g. 256)")
    ap.add_argument("--ledger", default="data/index/ledger.json")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    if a.backend == "fake":
        be = FakeBackend()
    elif a.backend == "local":
        be = LocalSTBackend(a.model, truncate_dim=a.truncate_dim)
    else:
        be = OpenRouterEmbedBackend(a.model, ledger=Ledger(a.ledger, a.max_cost))
    idx = build(a.vocab, be, a.index_dir, a.batch_size)
    print(f"index '{be.name}': {idx.vectors.shape} from_cache={idx.from_cache}")
