import pytest

from scripts.benchmark.significance import (
    N_RESAMPLES,
    bootstrap_ci,
    compare_paired,
    randomization_p_value,
)


def test_identical_runs_are_not_different():
    scores = [0.2, 0.5, 1.0, 0.0, 0.7]
    result = compare_paired(scores, scores)
    assert result["mean_diff"] == 0.0
    assert result["p_value"] == 1.0
    assert (result["ci_low"], result["ci_high"]) == (0.0, 0.0)
    assert (result["wins"], result["ties"], result["losses"]) == (0, 5, 0)


def test_consistent_improvement_gets_smallest_p_value():
    b = [i / 100 for i in range(50)]
    a = [x + 0.1 for x in b]
    result = compare_paired(a, b)
    assert result["mean_diff"] == pytest.approx(0.1)
    # Only the all-positive and all-negative sign patterns (probability 2 / 2**50)
    # reach the observed magnitude, so p sits at its floor 1 / (N + 1).
    assert result["p_value"] == 1 / (N_RESAMPLES + 1)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)
    assert result["wins"] == 50


def test_symmetric_differences_are_not_significant():
    diffs = [0.3, -0.3] * 40
    assert randomization_p_value(diffs) > 0.5
    low, high = bootstrap_ci(diffs)
    assert low < 0.0 < high


def test_results_are_deterministic_for_a_seed():
    diffs = [0.05, -0.2, 0.0, 0.4, 0.1, -0.05, 0.3]
    assert randomization_p_value(diffs, seed=7) == randomization_p_value(diffs, seed=7)
    assert bootstrap_ci(diffs, seed=7) == bootstrap_ci(diffs, seed=7)


def test_p_value_is_a_valid_probability():
    p = randomization_p_value([0.1, -0.4, 0.2], n_resamples=N_RESAMPLES)
    assert 1 / (N_RESAMPLES + 1) <= p <= 1.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        compare_paired([0.1], [0.1, 0.2])
