"""Bootstrap confidence intervals + paired significance testing (pure numpy)."""
import numpy as np

ALPHA = 0.05


def bootstrap_mean_ci(values, n_boot: int = 1000, alpha: float = ALPHA,
                      seed: int = 0):
    """95% CI for the mean of per-record values (recall@k, p@k, mrr, macro-F1).
    Returns (point, lo, hi). Vectorized resampling over records."""
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def _micro(tp, pred_n, gt_n, metric):
    p = np.divide(tp, pred_n, out=np.zeros_like(tp, dtype=float), where=pred_n > 0)
    r = np.divide(tp, gt_n, out=np.zeros_like(tp, dtype=float), where=gt_n > 0)
    if metric == "p":
        return p
    if metric == "r":
        return r
    denom = p + r
    return np.divide(2 * p * r, denom, out=np.zeros_like(p), where=denom > 0)


def bootstrap_micro_ci(counts, metric: str, n_boot: int = 1000,
                       alpha: float = ALPHA, seed: int = 0):
    """95% CI for micro P/R/F1 from per-record (tp, pred_n, gt_n) triples."""
    arr = np.asarray(counts, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return (0.0, 0.0, 0.0)
    tot = arr.sum(axis=0)
    point = float(_micro(np.array([tot[0]]), np.array([tot[1]]),
                         np.array([tot[2]]), metric)[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    sums = arr[idx].sum(axis=1)
    stats = _micro(sums[:, 0], sums[:, 1], sums[:, 2], metric)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def paired_randomization_pvalue(a, b, n_iter: int = 10000, seed: int = 0):
    """Two-sided paired approximate-randomization test on per-record values.
    H0: a and b come from the same system (per-record sign of (a-b) is random)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.shape[0]
    if n == 0:
        return 1.0
    observed = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_iter, n))
    means = np.abs((signs * d).mean(axis=1))
    return float((np.sum(means >= observed) + 1) / (n_iter + 1))
