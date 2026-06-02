# src/lcsh_benchmark/scaleup/assemble.py
"""Join one selected record's manifest GT/graded + corpus inputs/tags into the
canonical v2 dev/test schema. Pure functions, no I/O."""
from ..clean import clean_headings, is_marc_artifact
from ..consensus import merge_union
from ..normalize import normalize_label
from ..typing import type_by_tag

_INPUT_FIELDS = ("title", "title_vernacular", "authors", "authors_vernacular",
                 "date", "publisher", "physical_description", "abstract", "toc", "notes")
_LIST_FIELDS = {"authors", "authors_vernacular"}
_TAG_PRIORITY = ("650", "651", "655", "600", "610", "611", "630")
_LANG_NAMES = {
    "eng": "English", "ger": "German", "fre": "French", "spa": "Spanish",
    "rus": "Russian", "chi": "Chinese", "jpn": "Japanese", "ita": "Italian",
    "ara": "Arabic", "kor": "Korean", "por": "Portuguese", "pol": "Polish",
    "heb": "Hebrew", "tur": "Turkish", "hin": "Hindi",
}


def _size(v) -> int:
    return len(v) if isinstance(v, list) else len(v or "")


def merge_inputs(per_source: dict[str, dict]) -> tuple[dict, dict]:
    """Per field, take the longest non-empty value across sources (deterministic
    tie-break by source name). Returns (merged_fields, field_provenance)."""
    merged, prov = {}, {}
    for field in _INPUT_FIELDS:
        empty = [] if field in _LIST_FIELDS else ""
        best_val = empty
        best_src = ""
        for src in sorted(per_source):
            val = per_source[src].get(field) or empty
            if _size(val) > _size(best_val):
                best_val, best_src = val, src
        merged[field] = best_val
        if best_src:
            prov[field] = best_src
    return merged, prov


def tag_for_heading(heading: str, per_source_tagged: dict[str, dict]) -> str:
    """Which 6xx tag this heading appears under across sources (priority order)."""
    tags = set()
    for tagged in per_source_tagged.values():
        for tag, hs in tagged.items():
            if heading in hs:
                tags.add(tag)
    for t in _TAG_PRIORITY:
        if t in tags:
            return t
    return next(iter(sorted(tags)), "")


def assemble_record(key: str, manifest_rec: dict, per_source: dict[str, dict]) -> dict | None:
    cleaned: dict[str, list[str]] = {}
    for src, headings in manifest_rec["ground_truth_lcsh"].items():
        kept, _ = clean_headings(headings)
        if kept:
            cleaned[src] = kept
    merged = merge_union(cleaned)
    if not merged:
        return None

    # Unanimous-core GT: surfaces every holding source assigned (graded tier),
    # excluding MARC artifacts. The merged_norm intersection is defense-in-depth
    # (a legitimate non-artifact unanimous heading is always in the cleaned merge).
    merged_norm = {normalize_label(h) for h in merged}
    unanimous = sorted(
        {v["surface"] for v in manifest_rec.get("consensus_exact", {}).values()
         if v.get("tier") == "unanimous" and not is_marc_artifact(v["surface"])
         and normalize_label(v["surface"]) in merged_norm},
        key=str.lower)

    per_source_tagged = {s: r.get("headings", {}) for s, r in per_source.items()}
    heading_types = {h: type_by_tag(h, tag_for_heading(h, per_source_tagged)) for h in merged}

    inputs, prov = merge_inputs(per_source)
    oclc = next((r.get("oclc") for r in per_source.values() if r.get("oclc")), "")
    lccn = next((r.get("lccn") for r in per_source.values() if r.get("lccn")), "")
    lang = manifest_rec["lang"]
    return {
        "id": key, "oclc": oclc, "lccn": lccn,
        "language_code": lang, "language": _LANG_NAMES.get(lang, lang),
        "lc_class": manifest_rec["lc_class"],
        **inputs,
        "field_provenance": prov,
        "catalogs": sorted(cleaned),
        "catalog_count": len(cleaned),
        "n_agencies": manifest_rec.get("n_agencies", 0),
        "ground_truth_lcsh": cleaned,
        "ground_truth_lcsh_merged": merged,
        "ground_truth_lcsh_unanimous": unanimous,
        "heading_types": heading_types,
    }
