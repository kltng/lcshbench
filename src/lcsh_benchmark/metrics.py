# src/lcsh_benchmark/metrics.py
"""Pure scoring metrics for the LCSH benchmark (no I/O).

Match modes:
- exact: full normalized heading (normalize_label)
- root:  base only (subdivisions stripped)
Reuses lcsh_benchmark.normalize.normalize_label so scores match the Phase 1
hashes and the lcsh-onnx system.
"""
from collections.abc import Iterable

from .normalize import normalize_label


def norm_key(h: str) -> str:
    return normalize_label(h)


def root_key(h: str) -> str:
    return normalize_label(h).split("--", 1)[0]


def keys(headings: Iterable[str], mode: str) -> list[str]:
    f = root_key if mode == "root" else norm_key
    return [f(h) for h in headings]


def prf(gt: set[str], pred: set[str]) -> tuple[float, float, float]:
    tp = len(gt & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def micro_prf(per_record: list[tuple[set[str], set[str]]]) -> tuple[float, float, float]:
    tp = pred_n = gt_n = 0
    for gt, pred in per_record:
        tp += len(gt & pred)
        pred_n += len(pred)
        gt_n += len(gt)
    p = tp / pred_n if pred_n else 0.0
    r = tp / gt_n if gt_n else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def recall_at_k(gt: set[str], ranked: list[str], k: int) -> float:
    return len(gt & set(ranked[:k])) / max(1, len(gt))


def precision_at_k(gt: set[str], ranked: list[str], k: int) -> float:
    return len(gt & set(ranked[:k])) / k


def r_precision(gt: set[str], ranked: list[str]) -> float:
    n = len(gt)
    return len(gt & set(ranked[:n])) / max(1, n)


def mrr(gt: set[str], ranked: list[str]) -> float:
    for i, h in enumerate(ranked, 1):
        if h in gt:
            return 1.0 / i
    return 0.0
