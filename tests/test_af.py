"""Tests for irregular-rhythm (AF) detection."""

from __future__ import annotations

import numpy as np
import pytest

from ecgmon.analysis.af import (
    af_burden,
    af_episodes,
    assess_af_window,
    detect_af,
    normalised_rmssd,
    rr_entropy,
    turning_point_ratio,
)

FS = 360.0


def regular_peaks(n=200, rr_s=1.0, fs=FS):
    return (np.arange(n) * rr_s * fs).astype(int)


def irregular_peaks(n=200, rr_s=1.0, fs=FS, jitter=0.25, seed=0):
    rng = np.random.default_rng(seed)
    intervals = rng.uniform(rr_s * (1 - jitter), rr_s * (1 + jitter), n)
    return np.cumsum(intervals * fs).astype(int)


# ------------------------------------------------------------------ measures


def test_rmssd_of_constant_rhythm_is_zero():
    assert normalised_rmssd(np.ones(50)) == pytest.approx(0.0)


def test_rmssd_is_rate_independent():
    """Doubling heart rate must not change an irregularity measure."""
    rng = np.random.default_rng(0)
    rr = rng.uniform(0.8, 1.2, 100)
    assert normalised_rmssd(rr) == pytest.approx(normalised_rmssd(rr / 2), rel=1e-9)


def test_rmssd_higher_for_irregular():
    rng = np.random.default_rng(1)
    regular = np.full(100, 1.0) + rng.normal(0, 0.005, 100)
    irregular = rng.uniform(0.6, 1.4, 100)
    assert normalised_rmssd(irregular) > normalised_rmssd(regular)


def test_entropy_of_constant_is_zero():
    assert rr_entropy(np.ones(50)) == pytest.approx(0.0)


def test_entropy_higher_for_spread_intervals():
    rng = np.random.default_rng(2)
    tight = np.full(200, 1.0) + rng.normal(0, 0.01, 200)
    spread = rng.uniform(0.5, 1.5, 200)
    assert rr_entropy(spread) > rr_entropy(tight)


def test_entropy_is_bounded():
    rng = np.random.default_rng(3)
    assert 0.0 <= rr_entropy(rng.uniform(0.5, 1.5, 500)) <= 1.0


def test_turning_point_ratio_of_monotonic_series_is_zero():
    assert turning_point_ratio(np.arange(50, dtype=float)) == 0.0


def test_turning_point_ratio_of_alternating_is_one():
    x = np.array([1.0, 2.0] * 25)
    assert turning_point_ratio(x) == pytest.approx(1.0)


def test_turning_point_ratio_of_random_near_two_thirds():
    rng = np.random.default_rng(4)
    tpr = turning_point_ratio(rng.random(4000))
    assert 0.6 < tpr < 0.73


def test_measures_handle_short_input():
    for fn in (normalised_rmssd, rr_entropy, turning_point_ratio):
        assert fn(np.array([1.0])) == 0.0


# -------------------------------------------------------------------- window


def test_regular_rhythm_not_flagged():
    rr = np.full(40, 1.0)
    assert not assess_af_window(rr).is_af


def test_irregular_rhythm_flagged():
    rng = np.random.default_rng(5)
    rr = rng.uniform(0.5, 1.5, 60)
    assert assess_af_window(rr).is_af


def test_score_in_unit_range():
    rng = np.random.default_rng(6)
    for _ in range(10):
        rr = rng.uniform(0.3, 2.0, 50)
        assert 0.0 <= assess_af_window(rr).score <= 1.0


def test_sinus_variation_alone_is_not_af():
    """Gentle respiratory variation should not trip the detector."""
    t = np.arange(60)
    rr = 1.0 + 0.04 * np.sin(2 * np.pi * t / 12)   # ~4% smooth modulation
    assert not assess_af_window(rr).is_af


# ------------------------------------------------------------------ scanning


def test_detect_af_on_regular_recording():
    wins = detect_af(regular_peaks(n=300), FS, window_s=30.0)
    assert len(wins) > 0
    assert not any(w.is_af for w in wins)


def test_detect_af_on_irregular_recording():
    wins = detect_af(irregular_peaks(n=300, jitter=0.35), FS, window_s=30.0)
    assert any(w.is_af for w in wins)


def test_detect_af_too_few_beats():
    assert detect_af(np.array([0, 360]), FS) == []


def test_excluded_beats_do_not_create_false_irregularity():
    """Dropping a beat must not merge its two intervals into one long one.

    This was a real defect: removing ectopic beats and then differencing
    manufactured exactly the irregularity the exclusion was meant to remove,
    and PVC-heavy records scored as almost entirely AF.
    """
    peaks = regular_peaks(n=300, rr_s=1.0)
    exclude = np.zeros(peaks.size, dtype=bool)
    exclude[::7] = True          # scatter "ectopic" beats through a regular rhythm

    wins = detect_af(peaks, FS, window_s=30.0, exclude=exclude)
    assert len(wins) > 0
    assert not any(w.is_af for w in wins), "regular rhythm flagged after exclusion"


def test_exclusion_mask_of_wrong_length_is_ignored():
    peaks = regular_peaks(n=100)
    wins = detect_af(peaks, FS, window_s=30.0, exclude=np.zeros(5, dtype=bool))
    assert len(wins) > 0


# ----------------------------------------------------------------- reporting


def test_af_burden_of_no_windows():
    assert af_burden([]) == 0.0


def test_af_burden_fraction():
    wins = detect_af(irregular_peaks(n=400, jitter=0.4), FS, window_s=30.0)
    b = af_burden(wins)
    assert 0.0 <= b <= 1.0


def test_episodes_merge_adjacent_windows():
    wins = detect_af(irregular_peaks(n=600, jitter=0.4), FS, window_s=30.0)
    eps = af_episodes(wins, min_duration_s=30.0)
    for start, end in eps:
        assert end > start


def test_short_episodes_are_dropped():
    """A single flagged window is more often ectopy than a real paroxysm."""
    wins = detect_af(irregular_peaks(n=400, jitter=0.4), FS, window_s=30.0)
    long_only = af_episodes(wins, min_duration_s=10_000.0)
    assert long_only == []
