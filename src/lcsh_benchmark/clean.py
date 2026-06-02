# src/lcsh_benchmark/clean.py
"""Drop MARC-artifact strings that leaked into ground-truth headings.

Strings whose base segment looks like a MARC field tag (e.g. '653-11/',
'880-12') are extraction errors, not real LCSH headings.
"""
import re

# base (before the first --) starts with 3 digits then - or /
_ARTIFACT_RE = re.compile(r"^\d{3}[-/]")


def is_marc_artifact(heading: str) -> bool:
    base = heading.split("--", 1)[0].strip()
    return bool(_ARTIFACT_RE.match(base))


def clean_headings(headings: list[str]) -> tuple[list[str], list[str]]:
    kept, dropped = [], []
    for h in headings:
        (dropped if is_marc_artifact(h) else kept).append(h)
    return kept, dropped
