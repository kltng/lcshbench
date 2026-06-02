"""LLM run-to-run variance probe: aggregate a metric across repeated runs of the
same system (temperature > 0). The aggregation is pure + tested here; the actual
repeated paid runs are driven in W3 and fed in as scored dicts."""
import statistics


def variance_across_runs(runs: list[dict]) -> dict:
    """runs: list of {metric: value}. Returns {metric: {mean, std, n}}.

    `std` is the POPULATION standard deviation (statistics.pstdev): the repeated
    runs are treated as the complete set of observed runs, not a sample of a
    larger hypothetical distribution. 0.0 when n < 2."""
    keys = set().union(*[r.keys() for r in runs]) if runs else set()
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        out[k] = {"mean": statistics.fmean(vals),
                  "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                  "n": len(vals)}
    return out
