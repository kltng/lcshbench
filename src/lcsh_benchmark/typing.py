# src/lcsh_benchmark/typing.py
"""Type each GT heading by its authority source via the local lcsh.db.

The seed GT lost MARC field tags, so we type by which authority file the
heading's BASE resolves to: lcsh (subject) / lcnaf (name) / lcgft (genre) /
unmatched. The topical core = base resolves to 'lcsh'. Precise 650-vs-651 is
deferred to the v2 MARC re-harvest.
"""
import sqlite3
from collections.abc import Callable

from .normalize import normalize_label


def base_of(heading: str) -> str:
    return heading.split("--", 1)[0].strip()


def type_heading(heading: str, lookup: Callable[[str], str | None]) -> dict:
    base_norm = normalize_label(base_of(heading))
    authority = lookup(base_norm) or "unmatched"
    depth = heading.count("--")
    return {
        "base": base_norm,
        "authority": authority,
        "has_subdivision": depth > 0,
        "subdivision_depth": depth,
    }


def make_db_lookup(conn: sqlite3.Connection) -> Callable[[str], str | None]:
    """Return a lookup(normalized_base) -> authority|None backed by lcsh.db.

    Resolves to a single authority per normalized label; prefers 'lcsh' when a
    label exists in multiple authorities (topical reading wins).
    """
    rank = {"lcsh": 0, "lcgft": 1, "lcnaf": 2}

    def lookup(base_norm: str) -> str | None:
        rows = conn.execute(
            "SELECT DISTINCT authority FROM auth WHERE label_normalized = ?",
            (base_norm,),
        ).fetchall()
        auths = [r[0] for r in rows]
        if not auths:
            return None
        return sorted(auths, key=lambda a: rank.get(a, 9))[0]

    return lookup


_TAG_TYPE = {
    "650": "topical", "651": "geographic",
    "600": "name", "610": "name", "611": "name", "630": "name",
    "655": "genre",
}


def type_by_tag(heading: str, tag: str) -> dict:
    """Type a heading by its MARC 6xx field tag (no authority DB)."""
    depth = heading.count("--")
    return {
        "type": _TAG_TYPE.get(tag, "other"),
        "tag": tag,
        "base": normalize_label(base_of(heading)),
        "has_subdivision": depth > 0,
        "subdivision_depth": depth,
    }
