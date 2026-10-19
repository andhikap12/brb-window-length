"""reporting.py — cross-seed mean/std summary table for CV runs."""

import numpy as np


def summarize_cross_seed(
    per_seed_means: list[dict], metrics: list[tuple[str, str]]
) -> dict[str, tuple[float, float]]:
    """Compute and print (mean, std) per metric key across seeds.

    Returns {key: (mean, std)} using population std (np.std), matching the
    prior inline `_ms` logic. Prints the standard mean/std table.
    """
    out: dict[str, tuple[float, float]] = {}
    print(f"  {'Metric':<16s} {'Mean':>10s} {'Std':>10s}")
    print("  " + "-" * 40)
    for label, key in metrics:
        vals = [s[key] for s in per_seed_means]
        m, sd = float(np.mean(vals)), float(np.std(vals))
        out[key] = (m, sd)
        print(f"  {label:<16s} {m*100:9.2f}% {sd*100:9.2f}%")
    print("  " + "-" * 40)
    return out
