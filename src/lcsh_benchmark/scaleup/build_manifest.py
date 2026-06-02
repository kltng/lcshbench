# src/lcsh_benchmark/scaleup/build_manifest.py
"""Stream bulk MARC -> index (discard raw) -> graded consensus match -> sample."""
import argparse
import gzip
import json
import tarfile
from pathlib import Path

from pymarc.marcxml import map_xml

from . import composition as C
from .marc_index import index_record
from .match import match_consensus
from .sample import monograph_only, stratified_sample


def stream_index_marcxml_gz(gz_path: str, source: str, out_jsonl) -> int:
    """Columbia-style: a single gzipped MARCXML file."""
    count = {"n": 0}

    def h(rec):
        idx = index_record(rec, source)
        if idx["headings"]:
            out_jsonl.write(json.dumps(idx, ensure_ascii=False) + "\n")
            count["n"] += 1

    with gzip.open(gz_path, "rb") as fh:
        map_xml(h, fh)
    return count["n"]


def stream_index_marcxml_tar(tar_gz_path: str, source: str, out_jsonl) -> int:
    """Princeton-style: a .tar.gz of MARCXML member files."""
    count = {"n": 0}

    def h(rec):
        idx = index_record(rec, source)
        if idx["headings"]:
            out_jsonl.write(json.dumps(idx, ensure_ascii=False) + "\n")
            count["n"] += 1

    with tarfile.open(tar_gz_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is not None:
                map_xml(h, f)
    return count["n"]


def stream_index_marc_gz(gz_path: str, source: str, out_jsonl) -> int:
    """A single gzipped binary MARC21 file (e.g. LoC .utf8.gz / .mrc.gz)."""
    from pymarc import MARCReader
    count = {"n": 0}
    with gzip.open(gz_path, "rb") as fh:
        for rec in MARCReader(fh, to_unicode=True, permissive=True):
            if rec is None:
                continue
            idx = index_record(rec, source)
            if idx["headings"]:
                out_jsonl.write(json.dumps(idx, ensure_ascii=False) + "\n")
                count["n"] += 1
    return count["n"]


def stream_index_marc_tar(tar_gz_path: str, source: str, out_jsonl) -> int:
    """A .tar.gz of binary MARC21 member files (e.g. Harvard bibdata: 10 files)."""
    from pymarc import MARCReader
    count = {"n": 0}
    with tarfile.open(tar_gz_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            for rec in MARCReader(f, to_unicode=True, permissive=True):
                if rec is None:
                    continue
                idx = index_record(rec, source)
                if idx["headings"]:
                    out_jsonl.write(json.dumps(idx, ensure_ascii=False) + "\n")
                    count["n"] += 1
    return count["n"]


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build(index_paths: list[str], out: str) -> dict:
    records = [r for p in index_paths for r in load_jsonl(p)]
    matched = match_consensus(records, min_sources=2)
    books = monograph_only(matched)          # A3: select books only; corpus keeps all material
    core, breadth = stratified_sample(
        books, C.CORE_LANG_TARGETS, C.BREADTH_LANG_TARGETS, C.MUSIC_CAP, C.SEED)
    manifest = {"core": core, "breadth": breadth,
                "counts": {"matched": len(matched), "monographs": len(books),
                           "core": len(core), "breadth": len(breadth)}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return manifest["counts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", nargs="+", required=True, help="per-source JSONL index files")
    ap.add_argument("--out", default="data/raw/v2_manifest.json")
    args = ap.parse_args()
    print(build(args.indices, args.out))
