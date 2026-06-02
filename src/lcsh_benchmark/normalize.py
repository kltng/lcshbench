# src/lcsh_benchmark/normalize.py
"""Canonical normalization for LCSH heading comparison.

Ported from lcsh-onnx db-builder parser.normalize_label so the benchmark and
the system score headings identically.
"""
import re
import unicodedata

_DASH_RE = re.compile(r"\s*[—–]\s*")   # em-dash, en-dash -> --
_SEP_RE = re.compile(r"\s*--\s*")
_WS_RE = re.compile(r"\s+")


def normalize_label(label: str) -> str:
    s = unicodedata.normalize("NFC", label).lower()
    s = _DASH_RE.sub("--", s)
    s = _SEP_RE.sub("--", s)
    s = _WS_RE.sub(" ", s)
    return s.rstrip(".").strip()
