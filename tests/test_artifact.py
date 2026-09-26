"""Tests for motion-artefact detection."""

from __future__ import annotations

import numpy as np
import pytest

from ecgmon.io.record import ChannelRecord
from ecgmon.preprocessing.artifact import (
    amplitude_instability,
    assess_artifact,
    assess_artifact_window,
    beats_in_artifact,
    bsqi,
    kurtosis_score,
)

from test_pipeline import FS, synth_ecg


# ------------------------------------------------------------------ kurtosis


def test_clean_ecg_has_high_kurtosis():
    """A spiky ECG is heavy-tailed; Gaussian noise is not."""
    sig, _ = synth_ecg(duration_s=30.0)
    noise = np.random.default_rng(0).standard_normal(sig.size)
    assert kurtosis_score(sig) > kurtosis_score(noise)


def test_gaussian_noise_kurtosis_near_zero():
    noise = np.random.default_rng(1).standard_normal(20000)
    assert abs(kurtosis_score(noise)) < 0.5


def test_kurtosis_of_constant_signal_is_zero_not_nan():
    assert kurtosis_score(np.ones(1000)) == 0.0


def test_kurtosis_of_tiny_input():
    assert kurtosis_score(np.array([1.0, 2.0])) == 0.0


# ---------------------------------------------------------------------- bSQI


def test_bsqi_identical_detections_is_one():
    p = np.arange(0, 10000, 360)
    assert bsqi(p, p.copy(), FS) == pytest.approx(1.0)


def test_bsqi_disjoint_detections_is_zero():
    a = np.array([1000, 2000, 3000])
    b = np.array([1000, 2000, 3000]) + int(0.5 * FS)  # all outside tolerance
    assert bsqi(a, b, FS) == pytest.approx(0.0)


def test_bsqi_partial_agreement():
    a = np.array([1000, 2000, 3000, 4000])
    b = np.array([1000, 2000])          # agrees on 2, union is 4
    assert bsqi(a, b, FS) == pytest.approx(0.5)


def test_bsqi_tolerates_small_offsets():
    a = np.array([1000, 2000])
    b = a + int(0.05 * FS)              # 50 ms, inside the 150 ms tolerance
    assert bsqi(a, b, FS) == pytest.approx(1.0)


def test_bsqi_of_two_empty_detectors_is_one():
    """An empty window is not evidence of artefact."""
    assert bsqi(np.array([]), np.array([]), FS) == 1.0


def test_bsqi_one_empty_detector_is_zero():
    assert bsqi(np.array([1000]), np.array([]), FS) == 0.0


def test_bsqi_matches_each_beat_at_most_once():
    """Two detections near one reference must not both count as agreement."""
    a = np.array([1000, 1010])
    b = np.array([1005])
    # 1 match, union = 2 + 1 - 1 = 2
    assert bsqi(a, b, FS) == pytest.approx(0.5)


# ------------------------------------------------------------- instability


def test_steady_signal_has_low_instability():
    sig, _ = synth_ecg(duration_s=30.0)
    assert amplitude_instability(sig, FS) < 0.3


def test_amplitude_step_raises_instability():
    sig, _ = synth_ecg(duration_s=30.0)
    shifted = sig.copy()
    shifted[len(shifted) // 2:] *= 6.0      # abrupt amplitude change
    assert amplitude_instability(shifted, FS) > amplitude_instability(sig, FS)


def test_instability_of_flat_signal():
    assert amplitude_instability(np.zeros(10000), FS) == 1.0


def test_instability_of_short_input():
    assert amplitude_instability(np.zeros(10), FS) == 0.0


# -------------------------------------------------------------- windows


def test_clean_window_scores_low():
    sig, peaks = synth_ecg(duration_s=10.0)
    w = assess_artifact_window(sig, FS, peaks, peaks.copy())
    assert not w.is_artifact
    assert w.bsqi == pytest.approx(1.0)


def test_noise_window_scores_higher_than_clean():
    sig, peaks = synth_ecg(duration_s=10.0)
    noise = np.random.default_rng(2).standard_normal(sig.size) * 0.5

    clean = assess_artifact_window(sig, FS, peaks, peaks.copy())
    dirty = assess_artifact_window(noise, FS, np.array([100]), np.array([9000]))
    assert dirty.score > clean.score


def test_detector_disagreement_raises_score():
    """Same waveform, but the two detectors disagree -> more suspicious."""
    sig, peaks = synth_ecg(duration_s=10.0)
    agree = assess_artifact_window(sig, FS, peaks, peaks.copy())
    disagree = assess_artifact_window(sig, FS, peaks, peaks + int(0.4 * FS))
    assert disagree.score > agree.score


def test_assess_without_detectors_still_scores():
    sig, _ = synth_ecg(duration_s=10.0)
    w = assess_artifact_window(sig, FS)
    assert 0.0 <= w.score <= 1.0
    assert w.bsqi == 1.0


def test_scores_stay_in_unit_range():
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = rng.standard_normal(int(10 * FS)) * rng.uniform(0.1, 10)
        w = assess_artifact_window(x, FS)
        assert 0.0 <= w.score <= 1.0


# --------------------------------------------------------------- channels


def test_assess_artifact_covers_the_channel():
    sig, _ = synth_ecg(duration_s=60.0)
    ch = ChannelRecord("s1", FS, sig)
    windows = assess_artifact(ch, window_s=10.0, use_bsqi=False)
    assert len(windows) == 6
    assert windows[0].start_s == 0.0


def test_beats_in_artifact_returns_mask_per_beat():
    sig, peaks = synth_ecg(duration_s=60.0)
    ch = ChannelRecord("s1", FS, sig)
    mask = beats_in_artifact(peaks, ch, window_s=10.0)
    assert mask.shape == peaks.shape
    assert mask.dtype == bool


def test_beats_outside_signal_are_not_flagged():
    sig, peaks = synth_ecg(duration_s=30.0)
    ch = ChannelRecord("s1", FS, sig)
    beyond = np.array([sig.size + 1000, -5])
    assert not beats_in_artifact(beyond, ch).any()


def test_clean_channel_flags_few_beats():
    sig, peaks = synth_ecg(duration_s=60.0)
    ch = ChannelRecord("s1", FS, sig)
    assert beats_in_artifact(peaks, ch, window_s=10.0).mean() < 0.5
