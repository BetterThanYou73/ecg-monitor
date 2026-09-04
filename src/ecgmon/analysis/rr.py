"""RR-interval, heart-rate and heart-rate-variability analysis.

These are the features the dashboard's trend views are built from, and the
inputs to beat classification: most rhythm abnormalities are, at bottom,
statements about RR-interval patterns. Ectopic beats arrive early and are
followed by a compensatory pause, so they show up here before any morphology
analysis happens.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Interval bounds outside which a value is treated as a detection error
# rather than a real beat (24 bpm to 300 bpm).
MIN_RR_S = 0.20
MAX_RR_S = 2.50


@dataclass
class RRSeries:
    """RR intervals with the times at which they occur."""

    t_s: np.ndarray       # time of the interval's terminating beat, seconds
    rr_s: np.ndarray      # interval length in seconds
    valid: np.ndarray     # bool mask of physiologically plausible intervals

    @property
    def hr_bpm(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.rr_s > 0, 60.0 / self.rr_s, np.nan)

    @property
    def n_beats(self) -> int:
        return int(self.rr_s.size) + 1

    def clean(self) -> "RRSeries":
        """Drop implausible intervals."""
        return RRSeries(
            t_s=self.t_s[self.valid],
            rr_s=self.rr_s[self.valid],
            valid=np.ones(int(np.count_nonzero(self.valid)), dtype=bool),
        )


def rr_from_peaks(peaks: np.ndarray, fs: float) -> RRSeries:
    """Build an RR series from R-peak sample indices."""
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 2:
        empty = np.array([], dtype=float)
        return RRSeries(empty, empty, np.array([], dtype=bool))
    times = peaks / float(fs)
    rr = np.diff(times)
    valid = (rr >= MIN_RR_S) & (rr <= MAX_RR_S)
    return RRSeries(t_s=times[1:], rr_s=rr, valid=valid)


@dataclass
class HRVMetrics:
    """Time-domain HRV summary over a stretch of RR intervals."""

    n_intervals: int
    mean_hr_bpm: float
    min_hr_bpm: float
    max_hr_bpm: float
    mean_rr_ms: float
    sdnn_ms: float    # standard deviation of NN intervals: overall variability
    rmssd_ms: float   # root mean square of successive differences: short-term
    pnn50: float      # fraction of successive differences over 50 ms
    cv_rr: float      # coefficient of variation, useful as an AF hint

    def to_dict(self) -> dict:
        return asdict(self)


def hrv_metrics(series: RRSeries) -> HRVMetrics:
    """Time-domain HRV over the valid intervals of ``series``.

    Successive differences are taken only between intervals that are adjacent
    in the original recording; bridging a gap left by a rejected interval
    would invent a difference that never occurred.
    """
    idx = np.flatnonzero(series.valid)
    rr = series.rr_s[idx]
    if rr.size == 0:
        return HRVMetrics(0, *([float("nan")] * 8))

    rr_ms = rr * 1000.0
    adjacent = np.diff(idx) == 1
    diffs = np.diff(rr_ms)[adjacent] if rr_ms.size > 1 else np.array([])

    hr = 60.0 / rr
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else float("nan")
    pnn50 = float(np.mean(np.abs(diffs) > 50.0)) if diffs.size else float("nan")
    mean_rr = float(np.mean(rr_ms))

    return HRVMetrics(
        n_intervals=int(rr.size),
        mean_hr_bpm=float(np.mean(hr)),
        min_hr_bpm=float(np.min(hr)),
        max_hr_bpm=float(np.max(hr)),
        mean_rr_ms=mean_rr,
        sdnn_ms=float(np.std(rr_ms, ddof=1)) if rr.size > 1 else float("nan"),
        rmssd_ms=rmssd,
        pnn50=pnn50,
        cv_rr=float(np.std(rr_ms, ddof=1) / mean_rr) if rr.size > 1 and mean_rr else float("nan"),
    )


def heart_rate_trend(series: RRSeries, window_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Mean heart rate per fixed window.

    This is the 24-hour trend line on the dashboard. Windows containing no
    valid intervals yield NaN rather than being silently dropped, so gaps in
    a long recording remain visible instead of closing up.
    """
    clean = series.clean()
    if clean.rr_s.size == 0:
        return np.array([]), np.array([])

    end = float(clean.t_s[-1])
    edges = np.arange(0.0, end + window_s, window_s)
    centres, means = [], []
    hr = clean.hr_bpm
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (clean.t_s >= lo) & (clean.t_s < hi)
        centres.append((lo + hi) / 2.0)
        means.append(float(np.mean(hr[mask])) if np.any(mask) else np.nan)
    return np.array(centres), np.array(means)


def ectopic_candidates(series: RRSeries, prematurity: float = 0.80,
                       compensation: float = 1.15) -> np.ndarray:
    """Flag intervals matching the short-then-long ectopic signature.

    A premature ventricular or atrial beat truncates one interval and is
    followed by a compensatory pause. Comparing each interval against the
    local median rather than the global mean keeps this working while heart
    rate drifts over a long recording.

    This is a rhythm-level screen only. Confirming a beat as a PVC needs
    morphology, which is the next milestone; the intent here is to give that
    stage a small candidate set instead of every beat in 24 hours.

    Returns:
        Indices into ``series.rr_s`` of the *short* interval of each pair.
    """
    rr = series.rr_s
    if rr.size < 5:
        return np.array([], dtype=int)

    # Local median over roughly ten beats, tracking slow rate changes.
    k = min(11, rr.size if rr.size % 2 else rr.size - 1)
    if k < 3:
        return np.array([], dtype=int)
    pad = k // 2
    padded = np.pad(rr, pad, mode="edge")
    local = np.array([np.median(padded[i : i + k]) for i in range(rr.size)])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(local > 0, rr / local, np.nan)

    short = ratio < prematurity
    long_next = np.zeros_like(short)
    long_next[:-1] = ratio[1:] > compensation

    hits = np.flatnonzero(short & long_next & series.valid)
    return hits
