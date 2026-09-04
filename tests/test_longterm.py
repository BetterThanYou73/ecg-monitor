"""Tests for long-recording handling: dropouts, gaps and chunked processing.

These cover the failure modes that only appear at ambulatory scale, where a
recording is not uniformly good signal and cannot be held in memory whole.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecgmon.analysis.evaluation import score_detections
from ecgmon.detection.rpeaks import detect_rpeaks
from ecgmon.detection.streaming import (
    _dedupe,
    detect_rpeaks_chunked,
    process_channel_long,
)
from ecgmon.io.record import ChannelRecord
from ecgmon.preprocessing.filters import preprocess
from ecgmon.preprocessing.segments import (
    coverage,
    find_segments,
    usable_segments,
)

from test_pipeline import FS, synth_ecg


def make_long(duration_s=600.0, hr_bpm=60.0):
    sig, peaks = synth_ecg(duration_s=duration_s, hr_bpm=hr_bpm)
    return ChannelRecord("s1", FS, sig), peaks


# ------------------------------------------------------------- segmentation


def test_clean_signal_is_fully_usable():
    ch, _ = make_long(120.0)
    cov = coverage(ch)
    assert cov.fraction > 0.99
    assert cov.n_gaps == 0


def test_flatline_is_detected_as_dropout():
    ch, _ = make_long(120.0)
    sig = ch.signal.copy()
    sig[int(30 * FS) : int(50 * FS)] = 0.0  # 20 s detachment
    damaged = ChannelRecord("s1", FS, sig)

    bad = [s for s in find_segments(damaged) if not s.usable]
    assert len(bad) == 1
    assert bad[0].reason == "flatline"
    assert bad[0].duration_s == pytest.approx(20.0, abs=1.0)


def test_nan_gap_is_detected():
    ch, _ = make_long(120.0)
    sig = ch.signal.copy()
    sig[int(60 * FS) : int(75 * FS)] = np.nan
    damaged = ChannelRecord("s1", FS, sig)

    bad = [s for s in find_segments(damaged) if not s.usable]
    assert len(bad) == 1
    assert bad[0].reason == "nan"


def test_nan_does_not_disable_flatline_detection():
    """Regression: a NaN anywhere used to make every flatline undetectable.

    np.median over a diff array containing NaN returns NaN; comparisons
    against NaN are all False, so no sample was ever marked flat and a
    detached electrode elsewhere in the recording went unreported.
    """
    ch, _ = make_long(300.0)
    sig = ch.signal.copy()
    sig[int(50 * FS) : int(80 * FS)] = 0.0     # 30 s flatline
    sig[int(150 * FS) : int(165 * FS)] = np.nan  # 15 s gap
    damaged = ChannelRecord("s1", FS, sig)

    reasons = {s.reason for s in find_segments(damaged) if not s.usable}
    assert reasons == {"flatline", "nan"}, "both dropout kinds must be found"

    cov = coverage(damaged)
    # 45 s unusable out of 300 s
    assert cov.fraction == pytest.approx(0.85, abs=0.02)


def test_brief_dips_are_not_treated_as_dropouts():
    ch, _ = make_long(120.0)
    sig = ch.signal.copy()
    sig[int(30 * FS) : int(30 * FS) + int(0.2 * FS)] = 0.0  # 200 ms
    damaged = ChannelRecord("s1", FS, sig)
    assert all(s.usable for s in find_segments(damaged))


def test_short_usable_slivers_are_excluded():
    """A 3 s island between two dropouts is too short to analyse."""
    ch, _ = make_long(120.0)
    sig = ch.signal.copy()
    sig[int(20 * FS) : int(50 * FS)] = 0.0
    sig[int(53 * FS) : int(80 * FS)] = 0.0  # leaves a 3 s gap of good signal
    damaged = ChannelRecord("s1", FS, sig)

    for seg in usable_segments(damaged, min_duration_s=10.0):
        assert seg.duration_s >= 10.0


def test_coverage_of_entirely_flat_channel_is_zero():
    flat = ChannelRecord("s1", FS, np.zeros(int(60 * FS)))
    assert coverage(flat).fraction == pytest.approx(0.0, abs=0.02)


# --------------------------------------------------------------- chunking


def test_chunked_matches_whole_signal():
    ch, truth = make_long(300.0)
    cleaned = preprocess(ch.signal, FS)

    whole = detect_rpeaks(cleaned, FS)
    chunked = detect_rpeaks_chunked(cleaned, FS, chunk_s=30.0, overlap_s=5.0)

    agreement = score_detections(chunked, whole, FS)
    assert agreement.f1 > 0.99


def test_chunking_does_not_duplicate_beats_at_seams():
    ch, truth = make_long(300.0)
    cleaned = preprocess(ch.signal, FS)
    chunked = detect_rpeaks_chunked(cleaned, FS, chunk_s=30.0, overlap_s=5.0)

    # No two detections closer than the refractory period.
    assert np.min(np.diff(chunked)) >= int(0.20 * FS) - 1
    # And no beat lost: count should track the truth closely.
    score = score_detections(chunked, truth, FS)
    assert score.sensitivity > 0.97
    assert score.ppv > 0.97


def test_chunk_larger_than_signal_falls_back():
    ch, truth = make_long(30.0)
    cleaned = preprocess(ch.signal, FS)
    chunked = detect_rpeaks_chunked(cleaned, FS, chunk_s=600.0)
    whole = detect_rpeaks(cleaned, FS)
    assert np.array_equal(chunked, whole)


def test_chunked_on_empty_signal():
    assert detect_rpeaks_chunked(np.array([]), FS).size == 0


def test_dedupe_keeps_first_of_a_close_pair():
    peaks = np.array([100, 110, 1000])
    out = _dedupe(peaks, refractory=72)
    assert out.tolist() == [100, 1000]


# ------------------------------------------------------ full long pipeline


def test_process_channel_long_skips_dropouts():
    ch, truth = make_long(300.0)
    sig = ch.signal.copy()
    sig[int(100 * FS) : int(160 * FS)] = 0.0  # 60 s detachment
    damaged = ChannelRecord("s1", FS, sig)

    result = process_channel_long(damaged, chunk_s=60.0, powerline_hz=None)

    assert result["coverage"].fraction == pytest.approx(0.8, abs=0.05)
    # No beats should be reported from inside the dropout.
    peaks = result["peaks"]
    inside = peaks[(peaks >= int(100 * FS)) & (peaks < int(160 * FS))]
    assert inside.size == 0


def test_process_channel_long_reports_coverage_on_clean_signal():
    ch, truth = make_long(120.0)
    result = process_channel_long(ch, chunk_s=60.0, powerline_hz=None)
    assert result["coverage"].fraction > 0.99
    assert result["n_beats"] > 100
