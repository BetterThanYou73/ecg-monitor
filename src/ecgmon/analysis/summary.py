"""Longitudinal summaries for a monitoring session.

This is the layer the dashboard reads. It turns a beat list and per-beat
labels into the quantities a clinician or researcher actually looks at over
24 hours: how many abnormal beats, what fraction of the total, how they were
distributed through the recording, and what they looked like.

Every rate is reported against *analysed* time rather than elapsed time.
A recording where the electrode fell off for six hours has six fewer hours of
evidence, and a burden computed over elapsed time would understate it. The
coverage figure travels with the summary so the denominator is never implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.record import ChannelRecord
from ..preprocessing.segments import Coverage, coverage
from .rr import RRSeries, hrv_metrics, rr_from_peaks

SECONDS_PER_HOUR = 3600.0


@dataclass
class EventSummary:
    """Counts and rates for one class of abnormal beat."""

    label: str
    count: int
    burden_pct: float        # share of all analysed beats
    per_hour: float          # events per analysed hour
    max_hourly: int          # busiest single hour
    analysed_hours: float = 0.0
    hourly_counts: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    hour_edges: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def rate_is_extrapolated(self) -> bool:
        """True when per_hour projects beyond the evidence available.

        A 30-minute recording with 491 events yields "982 per hour", which
        reads as a measurement but is a doubling of what was observed. Short
        recordings must say so; a burden percentage stays honest either way
        because it is a ratio, not a projection.
        """
        return self.analysed_hours < 1.0

    def __str__(self) -> str:
        rate = f"{self.per_hour:.1f}/h"
        if self.rate_is_extrapolated:
            rate += f" (extrapolated from {self.analysed_hours:.2f} h)"
        return (
            f"{self.label}: {self.count} events, {self.burden_pct:.2f}% burden, "
            f"{rate} (peak hour {self.max_hourly})"
        )


@dataclass
class ChannelSummary:
    """Everything the dashboard shows for one sensor."""

    sensor_id: str
    position: str
    duration_s: float
    coverage: Coverage
    n_beats: int
    mean_hr: float
    min_hr: float
    max_hr: float
    sdnn_ms: float
    rmssd_ms: float
    events: dict = field(default_factory=dict)      # label -> EventSummary
    hr_trend_t: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    hr_trend_bpm: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    rr_t: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    rr_s: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    sqi_t: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    sqi: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def analysed_hours(self) -> float:
        return self.coverage.usable_s / SECONDS_PER_HOUR

    def text_report(self) -> str:
        lines = [
            f"{self.sensor_id} ({self.position})",
            f"  duration     {self.duration_s / SECONDS_PER_HOUR:.2f} h",
            f"  coverage     {self.coverage}",
            f"  beats        {self.n_beats}",
            f"  heart rate   {self.mean_hr:.1f} bpm "
            f"(min {self.min_hr:.1f}, max {self.max_hr:.1f})",
            f"  SDNN {self.sdnn_ms:.1f} ms   RMSSD {self.rmssd_ms:.1f} ms",
        ]
        for ev in self.events.values():
            lines.append(f"  {ev}")
        return "\n".join(lines)


def hourly_histogram(
    event_samples: np.ndarray, fs: float, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bin event times into whole hours.

    Returns ``(counts, edges_in_hours)``. The final bin covers a partial hour
    when the recording does not end on the hour; its count is not scaled up,
    because inflating a partial hour to a full-hour rate invents events.
    """
    n_hours = max(1, int(np.ceil(duration_s / SECONDS_PER_HOUR)))
    edges = np.arange(n_hours + 1, dtype=float)
    if event_samples.size == 0:
        return np.zeros(n_hours, dtype=int), edges
    hours = (np.asarray(event_samples, dtype=np.float64) / fs) / SECONDS_PER_HOUR
    counts, _ = np.histogram(hours, bins=edges)
    return counts.astype(int), edges


def summarize_events(
    label: str,
    event_samples: np.ndarray,
    total_beats: int,
    fs: float,
    duration_s: float,
    analysed_s: float,
) -> EventSummary:
    """Counts, burden and hourly distribution for one abnormal-beat class."""
    event_samples = np.asarray(event_samples)
    counts, edges = hourly_histogram(event_samples, fs, duration_s)
    n = int(event_samples.size)

    burden = 100.0 * n / total_beats if total_beats else 0.0
    hours = analysed_s / SECONDS_PER_HOUR
    per_hour = n / hours if hours > 0 else float("nan")

    return EventSummary(
        label=label,
        count=n,
        burden_pct=burden,
        per_hour=per_hour,
        max_hourly=int(counts.max()) if counts.size else 0,
        analysed_hours=hours,
        hourly_counts=counts,
        hour_edges=edges,
    )


def summarize_channel(
    ch: ChannelRecord,
    peaks: np.ndarray,
    event_samples: dict | None = None,
    sqi_windows: list | None = None,
    hr_window_s: float = 60.0,
) -> ChannelSummary:
    """Build the full summary for one sensor.

    Args:
        ch: The channel (raw or filtered; coverage is computed from it).
        peaks: Detected R-peak sample indices.
        event_samples: ``{label: sample_indices}`` for abnormal beats,
            e.g. ``{"PVC": np.array([...])}``.
        sqi_windows: Optional list of ``QualityWindow`` for the quality strip.
        hr_window_s: Averaging window for the heart-rate trend.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    cov = coverage(ch)

    series = rr_from_peaks(peaks, ch.fs)
    clean = series.clean()
    m = hrv_metrics(series)

    # Heart-rate trend, leaving NaN where a window held no valid intervals so
    # gaps stay visible rather than closing up.
    if clean.rr_s.size:
        from .rr import heart_rate_trend

        t_trend, hr_trend = heart_rate_trend(series, window_s=hr_window_s)
    else:
        t_trend, hr_trend = np.array([]), np.array([])

    events: dict = {}
    for label, samples in (event_samples or {}).items():
        events[label] = summarize_events(
            label, samples, int(peaks.size), ch.fs, ch.duration_s, cov.usable_s
        )

    if sqi_windows:
        sqi_t = np.array([w.start_s for w in sqi_windows])
        sqi_v = np.array([w.sqi for w in sqi_windows])
    else:
        sqi_t, sqi_v = np.array([]), np.array([])

    return ChannelSummary(
        sensor_id=ch.sensor_id,
        position=ch.position,
        duration_s=ch.duration_s,
        coverage=cov,
        n_beats=int(peaks.size),
        mean_hr=m.mean_hr_bpm,
        min_hr=m.min_hr_bpm,
        max_hr=m.max_hr_bpm,
        sdnn_ms=m.sdnn_ms,
        rmssd_ms=m.rmssd_ms,
        events=events,
        hr_trend_t=t_trend,
        hr_trend_bpm=hr_trend,
        rr_t=clean.t_s,
        rr_s=clean.rr_s,
        sqi_t=sqi_t,
        sqi=sqi_v,
    )


def representative_beats(
    x: np.ndarray,
    fs: float,
    event_samples: np.ndarray,
    n: int = 5,
    before_s: float = 0.6,
    after_s: float = 0.6,
) -> list:
    """Pick exemplar events, spread across the recording.

    Chosen at even intervals through the event list rather than at random, so
    the same recording always yields the same exemplars and they span the
    session instead of clustering in whichever stretch happened to be busiest.

    Returns a list of ``(centre_sample, time_axis_s, waveform)``.
    """
    x = np.asarray(x, dtype=np.float64)
    event_samples = np.sort(np.asarray(event_samples, dtype=np.int64))
    if event_samples.size == 0:
        return []

    n = min(n, event_samples.size)
    picks = event_samples[np.linspace(0, event_samples.size - 1, n).astype(int)]

    nb, na = int(round(before_s * fs)), int(round(after_s * fs))
    out = []
    for p in picks:
        lo, hi = int(p) - nb, int(p) + na
        if lo < 0 or hi >= x.size:
            continue
        t = (np.arange(lo, hi) - p) / fs
        out.append((int(p), t, x[lo:hi]))
    return out


def cross_sensor_agreement(
    event_by_sensor: dict, fs: float, tolerance_s: float = 0.15
) -> dict:
    """How consistently each abnormal event was seen across sensors.

    For a distributed multi-sensor system this is a central question: an event
    detected by every sensor is far more credible than one seen by a single
    electrode, which may be an artefact local to that site.

    Returns ``{"n_sensors": int, "consensus": {k: count}}`` where ``k`` is the
    number of sensors that saw an event and ``count`` how many events had that
    level of agreement.
    """
    sensors = list(event_by_sensor)
    if len(sensors) < 2:
        return {"n_sensors": len(sensors), "consensus": {}}

    tol = tolerance_s * fs
    # Use the first sensor's events as anchors, then count how many other
    # sensors reported an event within tolerance.
    anchors = np.sort(np.asarray(event_by_sensor[sensors[0]], dtype=np.int64))
    others = [np.sort(np.asarray(event_by_sensor[s], dtype=np.int64)) for s in sensors[1:]]

    consensus: dict = {}
    for a in anchors:
        k = 1
        for o in others:
            if o.size and np.min(np.abs(o - a)) <= tol:
                k += 1
        consensus[k] = consensus.get(k, 0) + 1

    return {"n_sensors": len(sensors), "consensus": consensus}
