# src/lcsh_benchmark/load.py
"""Load the two seed datasets and unify them to the canonical schema."""
import json

from .consensus import catalog_count

_STR_FIELDS = ("title", "language_code", "language", "date", "publisher",
               "physical_description", "abstract", "toc", "notes")


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["records"]


def to_canonical(rec: dict, source: str) -> dict:
    ids = rec.get("identifiers", {})
    gt = rec.get("ground_truth_lcsh", {}) or {}
    return {
        "id": f"{source}-{rec['id']}",
        "lccn": rec.get("lccn") or ids.get("lccn", ""),
        "isbn": rec.get("isbn") or ids.get("isbn", ""),
        "oclc": rec.get("oclc") or ids.get("oclc", ""),
        "authors": rec.get("authors", []),
        "genres": rec.get("genres", []),
        **{k: rec.get(k, "") for k in _STR_FIELDS},
        "catalogs": sorted(k for k, v in gt.items() if v),
        "catalog_count": catalog_count(gt),
        "ground_truth_lcsh": gt,
        "ground_truth_lcsh_merged": rec.get("ground_truth_lcsh_merged", []),
    }
