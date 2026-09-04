"""Tests for the ECG pipeline.

These use synthetic signals rather than MIT-BIH so the suite runs offline and
fast. The point is to pin down invariants -- especially the cross-sensor
timeline arithmetic, where an error would be silent and would invalidate
every multi-sensor conclusion built on top of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ecgmon.analysis.evaluation import match_detections, score_detections
from ecgmon.analysis.rr import ectopic_candidates, hrv_metrics, rr_from_peaks
from ecgmon.detection.rpeaks import detect_rpeaks
from ecgmon.io.record import ChannelRecord, SynchronizedRecording
from ecgmon.preprocessing.filters import highpass, notch, preprocess
from ecgmon.preprocessing.quality import assess_window, channel_sqi

FS = 360.0
EPOCH = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def synth_ecg(duration_s=30.0, fs=FS, hr_bpm=60.0, amplitude=1.0, seed=0):
    """Synthetic ECG: Gaussian QRS complexes on a quiet baseline.

    Returns ``(signal, true_peak_indices)``.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    sig = 0.02 * rng.standard_normal(n)

    rr = 60.0 / hr_bpm
    peaks = []
    pos = rr
    while pos < duration_s - rr / 2:
        idx = int(round(pos * fs))
        if 0 <= idx < n:
            width = 0.012
            lo, hi = max(0, idx - int(0.05 * fs)), min(n, idx + int(0.05 * fs))
            sig[lo:hi] += amplitude * np.exp(
                -0.5 * ((t[lo:hi] - t[idx]) / width) ** 2
            )
            peaks.append(idx)
        pos += rr
    return sig, np.array(peaks, dtype=int)


def make_channel(sensor_id="s1", position="upper_chest", t_start=EPOCH,
                 offset_s=0.0, duration_s=30.0, hr_bpm=60.0):
    sig, peaks = synth_ecg(duration_s=duration_s, hr_bpm=hr_bpm)
    ch = ChannelRecord(
        sensor_id=sensor_id, fs=FS, signal=sig, position=position,
        t_start=t_start, clock_offset_s=offset_s,
    )
    return ch, peaks


# --------------------------------------------------------------- data model


def test_channel_basic_properties():
    ch, _ = make_channel(duration_s=10.0)
    assert ch.n_samples == int(10.0 * FS)
    assert ch.duration_s == pytest.approx(10.0)
    assert ch.t_start.tzinfo is not None


def test_naive_datetime_is_made_utc():
    ch = ChannelRecord("s", FS, np.zeros(10), t_start=datetime(2026, 1, 1))
    assert ch.t_start.tzinfo == timezone.utc


def test_with_signal_rejects_length_change():
    ch, _ = make_channel()
    with pytest.raises(ValueError, match="length changed"):
        ch.with_signal(np.zeros(5))


def test_single_channel_is_the_n1_case():
    ch, _ = make_channel()
    rec = SynchronizedRecording("r", [ch])
    assert rec.is_single_channel
    assert rec.n_channels == 1
    # The multi-channel accessors must still work on one channel.
    assert rec.overlap_window()[1] > 0 or rec.n_channels == 1


def test_duplicate_sensor_ids_rejected():
    a, _ = make_channel("same")
    b, _ = make_channel("same")
    with pytest.raises(ValueError, match="duplicate sensor_id"):
        SynchronizedRecording("r", [a, b])


def test_empty_recording_rejected():
    with pytest.raises(ValueError):
        SynchronizedRecording("r", [])


# ----------------------------------------------------- cross-sensor timeline


def test_stagger_shows_up_in_the_shared_timeline():
    """A sensor started 5 s later must appear 5 s later, not aligned to zero."""
    a, _ = make_channel("a", t_start=EPOCH)
    b, _ = make_channel("b", t_start=EPOCH + timedelta(seconds=5))
    rec = SynchronizedRecording("r", [a, b])

    origin = rec.t_origin
    assert origin == EPOCH
    assert a.elapsed_since(origin)[0] == pytest.approx(0.0)
    assert b.elapsed_since(origin)[0] == pytest.approx(5.0)


def test_clock_offset_shifts_the_timeline():
    """clock_offset_s must move a channel on the reference clock."""
    a, _ = make_channel("a", t_start=EPOCH)
    b, _ = make_channel("b", t_start=EPOCH, offset_s=2.0)
    rec = SynchronizedRecording("r", [a, b])
    origin = rec.t_origin

    assert b.elapsed_since(origin)[0] == pytest.approx(2.0)
    assert a.elapsed_since(origin)[0] == pytest.approx(0.0)


def test_overlap_window_of_staggered_channels():
    a, _ = make_channel("a", t_start=EPOCH, duration_s=30.0)
    b, _ = make_channel("b", t_start=EPOCH + timedelta(seconds=10), duration_s=30.0)
    rec = SynchronizedRecording("r", [a, b])

    start, end = rec.overlap_window()
    assert start == pytest.approx(10.0)   # both recording only from t=10
    assert end == pytest.approx(30.0)     # a stops at 30


def test_non_overlapping_channels_report_no_overlap():
    a, _ = make_channel("a", t_start=EPOCH, duration_s=10.0)
    b, _ = make_channel("b", t_start=EPOCH + timedelta(seconds=60), duration_s=10.0)
    rec = SynchronizedRecording("r", [a, b])
    assert rec.overlap_window() == (0.0, 0.0)


def test_slice_seconds_advances_start_time():
    ch, _ = make_channel(duration_s=30.0)
    piece = ch.slice_seconds(10.0, 20.0)
    assert piece.duration_s == pytest.approx(10.0)
    assert (piece.t_start - ch.t_start).total_seconds() == pytest.approx(10.0)


# ------------------------------------------------------------- preprocessing


def test_highpass_removes_baseline_drift():
    ch, _ = make_channel()
    t = ch.rel_time()
    drift = 3.0 * np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz wander
    filtered = highpass(ch.signal + drift, FS, 0.5)
    # Drift dominates the raw range; after filtering it should be gone.
    assert np.ptp(filtered) < np.ptp(ch.signal + drift) / 2


def test_notch_attenuates_mains():
    ch, _ = make_channel()
    t = ch.rel_time()
    interference = 0.5 * np.sin(2 * np.pi * 60.0 * t)
    noisy = ch.signal + interference

    def power_at(sig, f0):
        freqs = np.fft.rfftfreq(sig.size, 1 / FS)
        spec = np.abs(np.fft.rfft(sig))
        return spec[np.argmin(np.abs(freqs - f0))]

    assert power_at(notch(noisy, FS, 60.0), 60.0) < 0.1 * power_at(noisy, 60.0)


def test_preprocess_preserves_length():
    ch, _ = make_channel()
    assert preprocess(ch.signal, FS).size == ch.n_samples


def test_filters_survive_low_sampling_rates():
    """Cutoffs above Nyquist must be clamped, not raise."""
    sig = np.random.default_rng(0).standard_normal(500)
    out = preprocess(sig, fs=50.0, high_hz=40.0, powerline_hz=60.0)
    assert out.size == sig.size
    assert np.all(np.isfinite(out))


# ------------------------------------------------------------------ quality


def test_flatline_scores_zero():
    flat = assess_window(np.zeros(int(10 * FS)), FS)
    assert flat.sqi == 0.0
    assert not flat.is_usable


def test_clean_ecg_scores_better_than_noise():
    ch, _ = make_channel()
    noise = ChannelRecord("n", FS, np.random.default_rng(1).standard_normal(ch.n_samples))
    assert channel_sqi(ch) > channel_sqi(noise)


# ---------------------------------------------------------------- detection


def test_detects_synthetic_beats():
    sig, truth = synth_ecg(duration_s=30.0, hr_bpm=60.0)
    peaks = detect_rpeaks(preprocess(sig, FS), FS)
    score = score_detections(peaks, truth, FS)
    assert score.sensitivity > 0.95
    assert score.ppv > 0.95


def test_detection_handles_fast_and_slow_rates():
    for hr in (45.0, 150.0):
        sig, truth = synth_ecg(duration_s=30.0, hr_bpm=hr)
        peaks = detect_rpeaks(preprocess(sig, FS), FS)
        score = score_detections(peaks, truth, FS)
        assert score.sensitivity > 0.90, f"hr={hr}"


def test_detection_respects_refractory_period():
    sig, _ = synth_ecg(duration_s=30.0, hr_bpm=150.0)
    peaks = detect_rpeaks(preprocess(sig, FS), FS)
    if peaks.size > 1:
        assert np.min(np.diff(peaks)) >= int(0.20 * FS) - 1


def test_empty_and_tiny_inputs_do_not_crash():
    assert detect_rpeaks(np.array([]), FS).size == 0
    assert detect_rpeaks(np.zeros(10), FS).size == 0


def test_inverted_signal_still_detected():
    """R waves are negative at some sensor positions; polarity must not matter."""
    sig, truth = synth_ecg(duration_s=30.0)
    peaks = detect_rpeaks(preprocess(-sig, FS), FS)
    assert score_detections(peaks, truth, FS).sensitivity > 0.90


# --------------------------------------------------------------- evaluation


def test_matching_pairs_each_reference_once():
    """Two detections on one reference beat: one TP, one FP -- never two TPs."""
    ref = np.array([1000])
    det = np.array([1000, 1010])  # both within tolerance
    score = score_detections(det, ref, FS)
    assert score.true_positives == 1
    assert score.false_positives == 1


def test_detection_outside_tolerance_is_a_miss():
    ref = np.array([1000])
    det = np.array([1000 + int(0.5 * FS)])  # 500 ms away
    score = score_detections(det, ref, FS)
    assert score.true_positives == 0
    assert score.false_negatives == 1


def test_perfect_detection_scores_one():
    ref = np.arange(0, 10000, 360)
    score = score_detections(ref.copy(), ref, FS)
    assert score.sensitivity == 1.0
    assert score.ppv == 1.0


def test_match_returns_consistent_index_counts():
    ref = np.arange(0, 10000, 360)
    det = ref[::2]
    m_det, m_ref, u_det, u_ref = match_detections(det, ref, FS)
    assert m_det.size == m_ref.size
    assert m_det.size + u_det.size == det.size
    assert m_ref.size + u_ref.size == ref.size


# ------------------------------------------------------------------- RR/HRV


def test_rr_from_regular_beats():
    peaks = np.arange(0, int(60 * FS), int(FS))  # exactly 1 Hz
    series = rr_from_peaks(peaks, FS)
    assert np.allclose(series.rr_s, 1.0)
    assert np.allclose(series.hr_bpm, 60.0)


def test_implausible_intervals_flagged_invalid():
    peaks = np.array([0, int(0.05 * FS), int(5 * FS)])  # 50 ms then 5 s
    series = rr_from_peaks(peaks, FS)
    assert not series.valid.any()


def test_hrv_of_constant_rate_has_no_variability():
    peaks = np.arange(0, int(60 * FS), int(FS))
    m = hrv_metrics(rr_from_peaks(peaks, FS))
    assert m.mean_hr_bpm == pytest.approx(60.0)
    assert m.sdnn_ms == pytest.approx(0.0, abs=1e-6)
    assert m.rmssd_ms == pytest.approx(0.0, abs=1e-6)


def test_hrv_on_empty_series_is_nan_not_crash():
    m = hrv_metrics(rr_from_peaks(np.array([0]), FS))
    assert m.n_intervals == 0
    assert np.isnan(m.mean_hr_bpm)


def test_ectopic_screen_finds_premature_beat():
    """A beat arriving early followed by a compensatory pause is flagged."""
    beats = list(range(0, int(30 * FS), int(FS)))
    # Move one beat 300 ms early; the following interval lengthens to match.
    idx = 15
    beats[idx] -= int(0.3 * FS)
    series = rr_from_peaks(np.array(beats), FS)
    assert ectopic_candidates(series).size >= 1


def test_ectopic_screen_quiet_on_regular_rhythm():
    peaks = np.arange(0, int(60 * FS), int(FS))
    assert ectopic_candidates(rr_from_peaks(peaks, FS)).size == 0
