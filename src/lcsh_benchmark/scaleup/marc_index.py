# src/lcsh_benchmark/scaleup/marc_index.py
"""Extract the compact selection index from a pymarc bibliographic record.

Keeps headings tagged by 6xx field tag so downstream typing can distinguish
650 topical / 651 geographic / 600-611 name / 655 genre. No I/O here.
"""
import re

_SUBJECT_TAGS = ("600", "610", "611", "630", "650", "651", "655")
# OCLC numbers appear as (OCoLC)123, (OCoLC)ocm00012345, (OCoLC)ocn123456789,
# or (OCoLC)on1234567890 — strip the optional ocm/ocn/on prefix + leading zeros.
_OCOLC = re.compile(r"\(OCoLC\)\s*(?:ocm|ocn|on)?0*(\d+)", re.IGNORECASE)


def _subfield(rec, tag, code):
    for f in rec.get_fields(tag):
        for v in f.get_subfields(code):
            if v and v.strip():
                return v.strip()
    return ""


def _oclc(rec):
    for f in rec.get_fields("035"):
        for v in f.get_subfields("a"):
            m = _OCOLC.search(v or "")
            if m and m.group(1).lstrip("0"):  # reject all-zero forms (the o:0 false-match bug)
                return m.group(1).lstrip("0")
    return ""


def _join(rec, tag, codes):
    parts = []
    for f in rec.get_fields(tag):
        for s in f.subfields:
            if s.code in codes and s.value and s.value.strip():
                parts.append(s.value.strip())
    return " ".join(parts)


def _all_a(rec, tags):
    """One $a per field, across the given tags (e.g. authors from 1xx + 7xx)."""
    out = []
    for tag in tags:
        for f in rec.get_fields(tag):
            for v in f.get_subfields("a"):
                if v and v.strip():
                    out.append(v.strip())
                    break
    return out


def _vern_a(rec, tags):
    """$a of 880 vernacular fields linked (via $6 "<tag>-<occ>") to the given tags.
    880 carries the original script for a romanized field (e.g. 245, 1xx)."""
    out = []
    for f in rec.get_fields("880"):
        link = (f.get_subfields("6") or [""])[0].split("-", 1)[0]
        if link in tags:
            for v in f.get_subfields("a"):
                if v and v.strip():
                    out.append(v.strip())
                    break
    return out


_NAME_TOPIC_GEO = {"600", "610", "611", "630", "650", "651"}
# Real LCC top-level classes. I/O/W/X/Y are NOT used (W=NLM, I=SuDoc, etc.).
_LCC_CLASSES = set("ABCDEFGHJKLMNPQRSTUVZ")


def _is_lcsh_heading(field) -> bool:
    """H2 source filter: keep only LCSH/LCGFT 6xx, drop MeSH/FAST/foreign/etc.

    Name/topical/geographic (600-651): ind2==0 (LCSH) or ind2==7 with $2=lcsh.
    Genre/form (655): $2=lcgft (LCGFT)."""
    src = (field.get_subfields("2") or [""])[0].strip().lower()
    if field.tag == "655":
        return src == "lcgft"
    if field.tag in _NAME_TOPIC_GEO:
        return field.indicator2 == "0" or (field.indicator2 == "7" and src == "lcsh")
    return False


# Subdivision subfields are joined with `--` for ALL 6xx tags.
_SUBDIVISION = {"v", "x", "y", "z"}
# Main-heading subfields per tag — joined with their own cataloging punctuation
# (a single space; the values already carry trailing commas/periods). Anything
# not listed here and not a subdivision (e.g. $0 $1 $2 $3 $4 $5 $6 $8) is dropped.
_MAIN_SUBFIELDS = {
    "650": {"a", "b"},                                # topical
    "651": {"a"},                                     # geographic
    "600": {"a", "b", "c", "d", "q", "t", "p", "n", "k", "l"},   # personal name (+ name-title)
    "610": {"a", "b", "c", "d", "q", "t", "p", "n", "k", "l"},   # corporate name
    "611": {"a", "b", "c", "d", "q", "t", "p", "n", "k", "l"},   # meeting name
    "630": {"a", "p", "n", "k", "l", "s"},            # uniform title
}


def _heading_text(field):
    """Reconstruct one LCSH/LCGFT heading string from a 6xx field.

    Contract (see tests/test_marc_index_a2_reconstruction.py):
      - Main-heading subfields (per `_MAIN_SUBFIELDS[field.tag]`) form the first
        segment, joined with a single space (values carry their own punctuation).
      - Each subdivision subfield ($v/$x/$y/$z, in document order) is its own
        segment.
      - All segments are joined with `--`. A heading with no main subfields
        (e.g. a bare 655 genre term, which has no entry in _MAIN_SUBFIELDS) is
        just its subdivision segments, or its $a.
      - Subfields that are neither main nor subdivision are dropped.
    """
    main_codes = _MAIN_SUBFIELDS.get(field.tag, {"a"})
    main_parts, segments = [], []
    for s in field.subfields:
        val = s.value.strip() if s.value else ""
        if not val:
            continue
        if s.code in main_codes:
            main_parts.append(val)
        elif s.code in _SUBDIVISION:
            segments.append(val)
    if main_parts:
        segments.insert(0, " ".join(main_parts))
    return "--".join(segments)


def index_record(rec, source: str) -> dict:
    f008 = rec["008"].data if rec.get("008") else ""
    lang = f008[35:38] if len(f008) >= 38 else ""
    ldr = str(rec.leader) if rec.leader else ""
    # Monograph: leader/06 type ∈ {a,t} (language material) AND leader/07 level == m.
    is_monograph = len(ldr) >= 8 and ldr[6] in ("a", "t") and ldr[7] == "m"
    cls_a = _subfield(rec, "050", "a") or _subfield(rec, "090", "a")
    first = cls_a[0].upper() if cls_a else ""
    lc_class = first if first in _LCC_CLASSES else ""  # reject non-LCC letters (I/O/W/X/Y)
    headings: dict[str, list[str]] = {}
    for f in rec.get_fields(*_SUBJECT_TAGS):
        if not _is_lcsh_heading(f):
            continue
        h = _heading_text(f)
        if h:
            headings.setdefault(f.tag, []).append(h)
    lccn = _subfield(rec, "010", "a")
    has_input = bool(rec.get_fields("520") or rec.get_fields("505"))
    agency = _subfield(rec, "040", "a") or _subfield(rec, "040", "d")
    abstract = _join(rec, "520", {"a", "b"})
    toc = _join(rec, "505", {"a", "t", "r"})
    notes = _join(rec, "500", {"a"})
    authors = _all_a(rec, ("100", "110", "111", "700", "710", "711"))[:10]
    date = f008[7:11] if len(f008) >= 11 and f008[7:11].isdigit() else (
        _subfield(rec, "264", "c") or _subfield(rec, "260", "c"))
    publisher = _subfield(rec, "264", "b") or _subfield(rec, "260", "b")
    physical_description = _join(rec, "300", {"a", "b", "c"})
    _AUTHOR_TAGS = ("100", "110", "111", "700", "710", "711")
    title_vernacular = next(iter(_vern_a(rec, ("245",))), "")
    authors_vernacular = _vern_a(rec, _AUTHOR_TAGS)[:10]
    return {
        "source": source,
        "is_monograph": is_monograph,
        "oclc": _oclc(rec),
        "lccn": lccn,
        "lang": lang.strip(),
        "lc_class": lc_class,
        "title": _subfield(rec, "245", "a"),
        "title_vernacular": title_vernacular,
        "headings": headings,
        "has_input": has_input,
        "agency": agency.upper(),
        "abstract": abstract,
        "toc": toc,
        "notes": notes,
        "authors": authors,
        "authors_vernacular": authors_vernacular,
        "date": date,
        "publisher": publisher,
        "physical_description": physical_description,
    }
