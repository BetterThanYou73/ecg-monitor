"""Tests for longitudinal summaries and dashboard generation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ecgmon.analysis.summary import (
    cross_sensor_agreement,
    hourly_histogram,
    representative_beats,
    summarize_channel,
    summarize_events,
)
from ecgmon.io.record import ChannelRecord, SynchronizedRecording
from ecgmon.viz.dashboard import build_dashboard

from test_pipeline import FS, synth_ecg


def make_channel(duration_s=120.0, sensor_id="s1"):
    sig, peaks = synth_ecg(duration_s=duration_s)
    return ChannelRecord(sensor_id, FS, sig), peaks


# ------------------------------------------------------------- histograms


def test_hourly_histogram_bins_by_hour():
    # events at 0.5 h and 1.5 h
    samples = np.array([int(0.5 * 3600 * FS), int(1.5 * 3600 * FS)])
    counts, edges = hourly_histogram(samples, FS, duration_s=2 * 3600)
    assert counts.tolist() == [1, 1]
    assert edges.size == 3


def test_hourly_histogram_with_no_events():
    counts, edges = hourly_histogram(np.array([]), FS, duration_s=3600)
    assert counts.sum() == 0
    assert counts.size == 1


def test_partial_final_hour_is_not_scaled_up():
    """A 90-minute recording has two bins; the second is a real half-hour."""
    samples = np.array([int(1.2 * 3600 * FS)])
    counts, _ = hourly_histogram(samples, FS, duration_s=1.5 * 3600)
    assert counts.size == 2
    assert counts[1] == 1  # counted once, not doubled to a full-hour rate


# ----------------------------------------------------------------- burden


def test_burden_is_share_of_beats():
    ev = summarize_events("PVC", np.arange(10), total_beats=100, fs=FS,
                          duration_s=3600, analysed_s=3600)
    assert ev.count == 10
    assert ev.burden_pct == pytest.approx(10.0)
    assert ev.per_hour == pytest.approx(10.0)


def test_rate_uses_analysed_time_not_elapsed():
    """Half the recording unusable means the rate doubles for the same count."""
    full = summarize_events("PVC", np.arange(10), 100, FS, 3600, analysed_s=3600)
    half = summarize_events("PVC", np.arange(10), 100, FS, 3600, analysed_s=1800)
    assert half.per_hour == pytest.approx(2 * full.per_hour)
    # Burden is a ratio over beats, so it must not move.
    assert half.burden_pct == pytest.approx(full.burden_pct)


def test_short_recording_flags_extrapolated_rate():
    ev = summarize_events("PVC", np.arange(100), 400, FS,
                          duration_s=1800, analysed_s=1800)
    assert ev.rate_is_extrapolated
    assert "extrapolated" in str(ev)


def test_long_recording_does_not_flag_extrapolation():
    ev = summarize_events("PVC", np.arange(100), 400, FS,
                          duration_s=7200, analysed_s=7200)
    assert not ev.rate_is_extrapolated
    assert "extrapolated" not in str(ev)


def test_zero_beats_does_not_divide_by_zero():
    ev = summarize_events("PVC", np.array([]), total_beats=0, fs=FS,
                          duration_s=3600, analysed_s=3600)
    assert ev.burden_pct == 0.0
    assert ev.count == 0


# ---------------------------------------------------------------- summary


def test_channel_summary_basic():
    ch, peaks = make_channel(duration_s=120.0)
    s = summarize_channel(ch, peaks, event_samples={"PVC": peaks[:5]})
    assert s.n_beats == peaks.size
    assert s.coverage.fraction > 0.95
    assert "PVC" in s.events
    assert s.events["PVC"].count == 5
    assert 50 < s.mean_hr < 70


def test_summary_reports_reduced_coverage_on_dropout():
    ch, peaks = make_channel(duration_s=200.0)
    sig = ch.signal.copy()
    sig[int(50 * FS):int(100 * FS)] = 0.0  # 50 s flatline
    damaged = ChannelRecord("s1", FS, sig)
    s = summarize_channel(damaged, peaks)
    assert s.coverage.fraction < 0.8


def test_text_report_mentions_events():
    ch, peaks = make_channel()
    s = summarize_channel(ch, peaks, event_samples={"PVC": peaks[:3]})
    assert "PVC" in s.text_report()


# ------------------------------------------------------- representative


def test_representative_beats_spread_across_recording():
    ch, peaks = make_channel(duration_s=120.0)
    picks = representative_beats(ch.signal, FS, peaks, n=5)
    assert len(picks) <= 5
    centres = [c for c, _, _ in picks]
    assert centres == sorted(centres)
    # Spread, not clustered at the start.
    assert centres[-1] - centres[0] > 0.5 * (peaks[-1] - peaks[0])


def test_representative_beats_is_deterministic():
    ch, peaks = make_channel()
    a = [c for c, _, _ in representative_beats(ch.signal, FS, peaks, n=4)]
    b = [c for c, _, _ in representative_beats(ch.signal, FS, peaks, n=4)]
    assert a == b


def test_representative_beats_with_no_events():
    ch, _ = make_channel()
    assert representative_beats(ch.signal, FS, np.array([]), n=5) == []


# ------------------------------------------------------- cross-sensor


def test_cross_sensor_agreement_counts_consensus():
    a = np.array([1000, 5000, 9000])
    b = np.array([1005, 5000])          # agrees on two
    result = cross_sensor_agreement({"a": a, "b": b}, FS)
    assert result["n_sensors"] == 2
    assert result["consensus"].get(2, 0) == 2   # two events seen by both
    assert result["consensus"].get(1, 0) == 1   # one seen by a only


def test_cross_sensor_agreement_single_sensor():
    result = cross_sensor_agreement({"a": np.array([1, 2])}, FS)
    assert result["consensus"] == {}


# --------------------------------------------------------------- dashboard


def test_dashboard_writes_self_contained_html(tmp_path: Path):
    ch, peaks = make_channel(duration_s=120.0)
    rec = SynchronizedRecording("test", [ch])
    summaries = {ch.sensor_id: summarize_channel(ch, peaks,
                                                 event_samples={"PVC": peaks[:4]})}
    out = build_dashboard(
        rec, summaries, {ch.sensor_id: peaks},
        event_label="PVC",
        examples=representative_beats(ch.signal, FS, peaks[:4], n=2),
        out_path=tmp_path / "d.html",
    )
    text = out.read_text(encoding="utf-8")
    assert out.exists()
    # Plotly must be inlined exactly once, or the file both bloats and breaks.
    assert text.count("plotly.js v") <= 1
    assert "Overview" in text
    assert "PVC" in text

    # The report must load nothing over the network. Checking for the substring
    # 'http' is not the test: the bundled plotting library contains URL strings
    # for map tiles and attribution that are never fetched unless a map is
    # drawn. What matters is that no *resource-loading tag* points outward.
    #
    # Each check is reduced to a bool before asserting. Asserting directly on a
    # multi-megabyte string makes pytest spend minutes rendering a failure
    # explanation over the whole document.
    external_script = bool(re.search(r'<script[^>]+src=["\']https?://', text))
    external_style = bool(re.search(r'<link[^>]+href=["\']https?://', text))
    external_iframe = bool(re.search(r'<iframe[^>]+src=["\']https?://', text))
    assert not external_script, "report loads a script over the network"
    assert not external_style, "report loads a stylesheet over the network"
    assert not external_iframe, "report embeds remote content"


def test_dashboard_handles_multiple_sensors(tmp_path: Path):
    a, peaks_a = make_channel(duration_s=60.0, sensor_id="a")
    b, peaks_b = make_channel(duration_s=60.0, sensor_id="b")
    rec = SynchronizedRecording("test", [a, b])
    summaries = {
        "a": summarize_channel(a, peaks_a, event_samples={"PVC": peaks_a[:2]}),
        "b": summarize_channel(b, peaks_b, event_samples={"PVC": peaks_b[:3]}),
    }
    out = build_dashboard(rec, summaries, {"a": peaks_a, "b": peaks_b},
                          out_path=tmp_path / "multi.html", inline_plotly=False)
    text = out.read_text(encoding="utf-8")
    assert "a" in text and "b" in text
    assert "Per-sensor comparison" in text


def test_dashboard_escapes_untrusted_text(tmp_path: Path):
    """Sensor ids and captions must not be able to inject markup."""
    sig, peaks = synth_ecg(duration_s=60.0)
    ch = ChannelRecord("<script>bad()</script>", FS, sig)
    rec = SynchronizedRecording("t", [ch])
    summaries = {ch.sensor_id: summarize_channel(ch, peaks)}
    out = build_dashboard(rec, summaries, {ch.sensor_id: peaks},
                          out_path=tmp_path / "esc.html", inline_plotly=False)
    text = out.read_text(encoding="utf-8")
    assert "<script>bad()</script>" not in text
    assert "&lt;script&gt;" in text
