# src/lcsh_benchmark/scaleup/run.py
"""Phased, resumable, parallel orchestrator for the production bulk pipeline.

Phases (each idempotent / re-runnable):
  acquire-index : parallel download -> parse -> per-chunk JSONL shard
                  (raw discarded, or kept under --cache-dir for re-extraction)
  load          : serial shards -> SQLite records
  match         : SQL prefilter + Python graded consensus -> consensus table + report
  sample        : stratified selection -> data/raw/v2_manifest.json
"""
import argparse
import functools
import json
import multiprocessing as mp
import os
import tempfile
import time
import urllib.request
from pathlib import Path

from . import composition as C
from .build_manifest import (load_jsonl, stream_index_marcxml_gz,
                             stream_index_marcxml_tar, stream_index_marc_gz,
                             stream_index_marc_tar)
from .match import match_consensus, match_key
from .sample import monograph_only, stratified_sample
from .sources import iter_chunks
from . import store

CORPUS_DIR = "data/raw/v2/corpus"
DB_PATH = "data/raw/v2/v2.db"


def index_chunk_to_shard(local_path: str, source: str, chunk_id: str,
                         container: str, corpus_dir: str) -> tuple[int, str]:
    shard = Path(corpus_dir) / source / f"{chunk_id}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with open(shard, "w", encoding="utf-8") as out:
        if container == "gz_marcxml":
            n = stream_index_marcxml_gz(local_path, source, out)
        elif container == "tar_marcxml":
            n = stream_index_marcxml_tar(local_path, source, out)
        elif container == "gz_marc":
            n = stream_index_marc_gz(local_path, source, out)
        elif container == "tar_marc":
            n = stream_index_marc_tar(local_path, source, out)
        else:
            raise ValueError(f"unknown container: {container}")
    return n, str(shard)


def _download(url: str, dest: str, retries: int = 4) -> None:
    last = None
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:   # network/timeout — retry with backoff
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # 1s, 2s, 4s
    raise last


def _process_chunk(desc: dict, corpus_dir: str, cache_dir: str | None = None) -> dict:
    """Worker: download -> index. Picklable (module-level).

    With a cache_dir, a remote download is kept at cache_dir/<source>/<chunk_id>
    and reused on re-runs (so re-extraction needs no re-download). Local file://
    chunks are already persistent, so they index from a temp copy and are never
    duplicated into the cache. Without a cache_dir, raw is discarded as before.
    """
    is_local = desc["url"].startswith("file:")
    if cache_dir and not is_local:
        raw = Path(cache_dir) / desc["source"] / desc["chunk_id"]
        raw.parent.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        fd, tmp = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        raw = Path(tmp)
        cleanup = True
    try:
        if not (raw.exists() and raw.stat().st_size > 0):   # cache miss / temp
            _download(desc["url"], str(raw))
        n, _ = index_chunk_to_shard(str(raw), desc["source"], desc["chunk_id"],
                                    desc["container"], corpus_dir)
        return {"source": desc["source"], "chunk_id": desc["chunk_id"],
                "status": "done", "n": n}
    except Exception as e:
        return {"source": desc["source"], "chunk_id": desc["chunk_id"],
                "status": "failed", "n": 0, "error": str(e)}
    finally:
        if cleanup and raw.exists():
            raw.unlink()


def run_acquire(cx, srcs: list[str], workers: int, corpus_dir: str,
                cache_dir: str | None = None) -> dict:
    descs = [d for s in srcs for d in iter_chunks(s)
             if not store.chunk_done(cx, s, d["chunk_id"])]
    done = failed = 0
    worker = functools.partial(_process_chunk, corpus_dir=corpus_dir, cache_dir=cache_dir)
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(worker, descs):
            store.mark_chunk(cx, res["source"], res["chunk_id"], res["status"], res["n"])
            done += res["status"] == "done"
            failed += res["status"] == "failed"
    return {"chunks": len(descs), "done": done, "failed": failed}


def _record_row(idx: dict, chunk_id: str) -> dict:
    return {"source": idx["source"], "chunk_id": chunk_id, "match_key": match_key(idx),
            "oclc": idx["oclc"], "lccn": idx["lccn"], "lang": idx["lang"],
            "lc_class": idx["lc_class"], "agency": idx.get("agency", ""),
            "has_input": int(bool(idx["has_input"])),
            "is_monograph": int(bool(idx.get("is_monograph"))),
            "n_headings": sum(len(v) for v in idx["headings"].values()),
            "title": idx["title"], "headings_json": json.dumps(idx["headings"], ensure_ascii=False)}


def load_shards(cx, corpus_dir: str) -> int:
    loaded = set(json.loads(store.get_meta(cx, "loaded_shards") or "[]"))
    total = 0
    for shard in sorted(Path(corpus_dir).rglob("*.jsonl")):
        sp = str(shard)
        if sp in loaded:
            continue
        store.delete_chunk_records(cx, shard.parent.name, shard.stem)
        rows = [_record_row(idx, shard.stem) for idx in load_jsonl(sp)]
        store.insert_records(cx, rows)
        total += len(rows)
        loaded.add(sp)
        store.set_meta(cx, "loaded_shards", json.dumps(sorted(loaded)))
    return total


def run_match(cx, min_sources: int = 2, min_agencies: int = 2) -> int:
    store.clear_consensus(cx)
    matched = []
    for key in store.iter_consensus_keys(cx, min_sources, min_agencies):
        matched += match_consensus(store.members_for_key(cx, key),
                                   min_sources=min_sources,
                                   require_independent_agencies=min_agencies)
    store.write_consensus(cx, matched)
    return len(matched)


def print_pool_report(cx) -> None:
    by_lang = store.counts_by_lang(cx)
    print("=== consensus pool (data-driven size decision) ===")
    print("by language:", by_lang)
    print("by LC class:", store.counts_by_class(cx))
    print("total:", sum(by_lang.values()))


def run_sample(cx, out: str, core: dict | None = None, breadth: dict | None = None,
               scale: float = 1.0) -> dict:
    core = core or {k: round(v * scale) for k, v in C.CORE_LANG_TARGETS.items()}
    breadth = breadth or {k: round(v * scale) for k, v in C.BREADTH_LANG_TARGETS.items()}
    rows = list(store.iter_consensus_rows(cx))
    books = monograph_only(rows)             # A3: select books only; corpus keeps all material
    print(f"[sample] monograph filter: {len(books)}/{len(rows)} kept "
          f"({len(rows) - len(books)} non-monographs dropped)")
    sel_core, sel_breadth = stratified_sample(books, core, breadth, C.MUSIC_CAP, C.SEED)
    manifest = {"core": sel_core, "breadth": sel_breadth,
                "counts": {"pool": len(rows), "monographs": len(books),
                           "core": len(sel_core), "breadth": len(sel_breadth)}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return manifest["counts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["acquire-index", "load", "match", "sample"])
    ap.add_argument("--sources", nargs="+",
                    default=["columbia", "princeton", "harvard", "loc"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--corpus", default=CORPUS_DIR)
    ap.add_argument("--out", default="data/raw/v2_manifest.json")
    ap.add_argument("--scale", type=float, default=1.0, help="2.0 ~= 10K target")
    ap.add_argument("--cache-dir", default=None,
                    help="keep raw remote downloads here (reused on re-runs; no re-download)")
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    cx = store.connect(args.db)
    if args.phase == "acquire-index":
        print(run_acquire(cx, args.sources, args.workers, args.corpus, args.cache_dir))
    elif args.phase == "load":
        print({"loaded": load_shards(cx, args.corpus)})
    elif args.phase == "match":
        print({"consensus": run_match(cx)})
        print_pool_report(cx)
    elif args.phase == "sample":
        print(run_sample(cx, args.out, scale=args.scale))
