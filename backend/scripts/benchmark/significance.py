"""Paired comparisons of two retrieval configurations over the same queries.

Both tests resample per-query score differences, so they make no normality assumption
and need no dependencies. Seeds are fixed so reports are reproducible.
"""

import random
from typing import TypedDict

N_RESAMPLES = 10_000


class PairedComparison(TypedDict):
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    wins: int
    ties: int
    losses: int


def randomization_p_value(
    diffs: list[float], n_resamples: int = N_RESAMPLES, seed: int = 0
) -> float:
    """Two-sided paired randomization (sign-flip) test of mean difference = 0."""
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        bits = rng.getrandbits(len(nonzero))
        total = sum(d if (bits >> i) & 1 else -d for i, d in enumerate(nonzero))
        if abs(total) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_resamples + 1)


def bootstrap_ci(
    diffs: list[float],
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean difference."""
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * n_resamples)]
    high = means[min(int((1.0 - tail) * n_resamples), n_resamples - 1)]
    return low, high


def compare_paired(a: list[float], b: list[float]) -> PairedComparison:
    """Compare per-query scores of configuration a against b (same query order)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired comparison needs two equal-length, non-empty lists")
    diffs = [x - y for x, y in zip(a, b)]
    ci_low, ci_high = bootstrap_ci(diffs)
    return {
        "mean_diff": sum(diffs) / len(diffs),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": randomization_p_value(diffs),
        "wins": sum(1 for d in diffs if d > 0),
        "ties": sum(1 for d in diffs if d == 0),
        "losses": sum(1 for d in diffs if d < 0),
    }
