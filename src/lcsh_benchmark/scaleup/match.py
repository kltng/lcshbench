# src/lcsh_benchmark/scaleup/match.py
"""Offline multi-source consensus: group bulk-MARC index records by shared
identifier and keep those with >=min_sources distinct assigning sources;
optionally require >=N distinct 040 cataloging agencies (independence)."""
from collections import defaultdict

from lcsh_benchmark.normalize import normalize_label


SOURCE_PRECEDENCE = ("harvard", "columbia", "princeton")


def pick_stratify_fields(members: list[dict]) -> dict:
    """Deterministically choose lang/lc_class for stratification: the value from
    the highest-precedence source that has a non-empty one (ties broken by the
    fixed SOURCE_PRECEDENCE), so the choice never depends on DB row order."""
    def by_prec(m):
        s = m.get("source", "")
        return SOURCE_PRECEDENCE.index(s) if s in SOURCE_PRECEDENCE else len(SOURCE_PRECEDENCE)
    ordered = sorted(members, key=by_prec)
    def first_nonempty(field):
        for m in ordered:
            if m.get(field):
                return m[field]
        return ""
    return {"lang": first_nonempty("lang"), "lc_class": first_nonempty("lc_class")}


def _key(rec: dict) -> str:
    return f"o:{rec['oclc']}" if rec.get("oclc") else (f"l:{rec['lccn']}" if rec.get("lccn") else "")


def match_key(rec: dict) -> str:
    """Public match-key: 'o:<oclc>' else 'l:<lccn>' else ''."""
    return _key(rec)


def _flatten(headings: dict[str, list[str]]) -> list[str]:
    return [h for tag in headings for h in headings[tag]]


def _grade(gt: dict[str, list[str]], root: bool) -> dict:
    """Per-heading cross-catalog agreement. key = normalized (root or exact) form;
    value = {surface, votes (# assigning sources), sources, tier}."""
    def keyfn(h: str) -> str:
        nk = normalize_label(h)
        return nk.split("--", 1)[0] if root else nk

    def surface(h: str) -> str:
        return h.split("--", 1)[0].strip() if root else h

    votes: dict[str, set] = defaultdict(set)
    first_surface: dict[str, str] = {}
    for source, headings in gt.items():
        for h in headings:
            k = keyfn(h)
            if not k:
                continue
            first_surface.setdefault(k, surface(h))
            votes[k].add(source)
    cc = len(gt)
    out = {}
    for k, srcs in votes.items():
        v = len(srcs)
        tier = "unanimous" if v == cc else ("single" if v == 1 else "majority")
        out[k] = {"surface": first_surface[k], "votes": v,
                  "sources": sorted(srcs), "tier": tier}
    return out


def match_consensus(records: list[dict], min_sources: int = 2,
                    require_independent_agencies: int = 0) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        k = _key(r)
        if k:
            groups[k].append(r)

    out = []
    for k, members in groups.items():
        gt = {}
        tagged: dict[str, list[str]] = defaultdict(list)
        agencies: set[str] = set()
        for m in members:
            flat = _flatten(m["headings"])
            if flat:
                gt[m["source"]] = sorted(set(gt.get(m["source"], [])) | set(flat))
                ag = (m.get("agency") or "").strip()
                if ag:
                    agencies.add(ag.upper())
            for tag, hs in m["headings"].items():
                tagged[tag].extend(hs)
        if len(gt) < min_sources:
            continue
        if len(agencies) < require_independent_agencies:
            continue
        first = members[0]
        strat = pick_stratify_fields(members)
        out.append({
            "key": k,
            "oclc": first.get("oclc", ""),
            "lccn": first.get("lccn", ""),
            "lang": strat["lang"],
            "lc_class": strat["lc_class"],
            "title": next((m["title"] for m in members if m["title"]), ""),
            "has_input": any(m["has_input"] for m in members),
            "is_monograph": any(m.get("is_monograph") for m in members),
            "ground_truth_lcsh": gt,
            "catalog_count": len(gt),
            "n_agencies": len(agencies),
            "tagged_headings": {t: sorted(set(v)) for t, v in tagged.items()},
            "consensus_exact": _grade(gt, root=False),
            "consensus_root": _grade(gt, root=True),
        })
    out.sort(key=lambda r: r["key"])
    return out
