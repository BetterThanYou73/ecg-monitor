"""Atrial fibrillation detection from RR-interval irregularity.

AF is a rhythm diagnosis, not a beat diagnosis, so unlike the beat
classifiers this works on windows of RR intervals rather than on individual
beats. Its defining feature is that ventricular response becomes irregularly
irregular: intervals vary without the pattern that ectopy or sinus arrhythmia
produce.

The classical evidence for AF also includes absent P waves and fibrillatory
baseline activity. Neither is used here. P-wave detection is unreliable on a
single channel -- the wave is small, and the attempt to use it for
supraventricular beat classification in this codebase did not work -- and
relying on a feature that fails silently is worse than not using it.
Irregularity alone is measurable and testable.

Three measures over each window:

* **Normalised RMSSD.** Beat-to-beat interval change, scaled by mean RR so it
  does not simply track heart rate.
* **Shannon entropy** of the RR histogram. AF spreads intervals across many
  values; regular rhythms concentrate them.
* **Turning-point ratio.** The fraction of intervals that are local extrema.
  For an independent random sequence this tends to 2/3; a regular rhythm
  falls well below it. This one is distribution-free, which matters because
  the other two depend on amplitude scaling that varies between subjects.

**What this actually detects.** Measured against MIT-BIH rhythm annotations
over 13 records and 783 windows, with ectopic beats excluded:

=================================  ======  ======  ==========
target                             Se      PPV     specificity
=================================  ======  ======  ==========
AFIB only                          98.9%   68.6%   74.9%
any irregular supraventricular     98.7%   78.4%   81.2%
=================================  ======  ======  ==========

The second row is the honest description. A third of what the first row calls
false positives are atrial flutter, atrial bigeminy or supraventricular
tachycardia -- rhythms that are genuinely irregular and genuinely abnormal,
just not fibrillation. Separating AF from those requires atrial activity,
which is what this deliberately does not use.

So: read a positive as "irregular supraventricular rhythm, AF among the
candidates", not as a diagnosis of AF.

Frequent ectopy is the other confound, and it is handled rather than
disclaimed. A run of PVCs produces RR irregularity these measures cannot
distinguish from AF, so beat classification runs first and ectopic beats are
excluded. Without that exclusion the PVC-heavy records score 96.7% and 98.3%
AF -- entirely wrong. With it they fall to 11.9% and 27.3%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Long enough for irregularity to be measurable, short enough to localise an
# episode. AF paroxysms shorter than this are reported as the window they
# fall in, not resolved within it.
DEFAULT_WINDOW_S = 30.0
MIN_INTERVALS = 8


@dataclass
class AFWindow:
    """AF assessment for one window of RR intervals."""

    start_s: float
    end_s: float
    n_intervals: int
    rmssd_norm: float
    entropy: float
    turning_point_ratio: float
    score: float

    @property
    def is_af(self) -> bool:
        return self.score >= 0.5

    def __repr__(self) -> str:
        return (
            f"AFWindow({self.start_s:.0f}-{self.end_s:.0f}s, score={self.score:.2f}, "
            f"rmssd={self.rmssd_norm:.3f}, H={self.entropy:.2f}, "
            f"tpr={self.turning_point_ratio:.2f})"
        )


def normalised_rmssd(rr: np.ndarray) -> float:
    """RMSSD divided by mean RR, so it measures irregularity not rate."""
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < 2:
        return 0.0
    mean = np.mean(rr)
    if mean <= 0:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)) / mean)


def rr_entropy(rr: np.ndarray, n_bins: int = 16) -> float:
    """Shannon entropy of the RR histogram, normalised to [0, 1].

    Binned over the window's own range rather than a fixed grid, so the
    measure reflects how intervals are spread within the rhythm rather than
    how fast the heart happens to be beating.
    """
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < 2:
        return 0.0
    lo, hi = np.min(rr), np.max(rr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    counts, _ = np.histogram(rr, bins=n_bins, range=(lo, hi))
    p = counts[counts > 0] / counts.sum()
    h = -np.sum(p * np.log(p))
    return float(h / np.log(n_bins))


def turning_point_ratio(rr: np.ndarray) -> float:
    """Fraction of interior points that are local maxima or minima.

    Approaches 2/3 for an independent random sequence and falls toward zero
    for a smooth or regular one. Being rank-based, it is unaffected by the
    amplitude scaling that differs between subjects.
    """
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < 3:
        return 0.0
    a, b, c = rr[:-2], rr[1:-1], rr[2:]
    turning = ((b > a) & (b > c)) | ((b < a) & (b < c))
    return float(np.count_nonzero(turning) / turning.size)


def assess_af_window(rr: np.ndarray, start_s: float = 0.0,
                     end_s: float | None = None) -> AFWindow:
    """Score one window of RR intervals (in seconds) for AF."""
    rr = np.asarray(rr, dtype=np.float64)
    rr = rr[np.isfinite(rr)]

    r = normalised_rmssd(rr)
    h = rr_entropy(rr)
    t = turning_point_ratio(rr)

    # Each measure mapped to 0-1 "AF-ness" against thresholds that separate
    # regular from irregular rhythm. Deliberately soft: the combination is
    # what decides, not any single crossing.
    s_rmssd = float(np.clip((r - 0.05) / 0.20, 0.0, 1.0))
    s_entropy = float(np.clip((h - 0.55) / 0.35, 0.0, 1.0))
    s_tpr = float(np.clip((t - 0.40) / 0.25, 0.0, 1.0))

    score = 0.45 * s_rmssd + 0.30 * s_entropy + 0.25 * s_tpr

    return AFWindow(
        start_s=start_s,
        end_s=end_s if end_s is not None else start_s + float(np.sum(rr)),
        n_intervals=int(rr.size),
        rmssd_norm=r,
        entropy=h,
        turning_point_ratio=t,
        score=float(np.clip(score, 0.0, 1.0)),
    )


def detect_af(
    peaks: np.ndarray,
    fs: float,
    window_s: float = DEFAULT_WINDOW_S,
    exclude: np.ndarray | None = None,
) -> list:
    """Scan a recording for AF, window by window.

    Args:
        peaks: R-peak sample indices.
        fs: Sampling rate.
        window_s: Window length in seconds.
        exclude: Optional boolean mask over ``peaks`` marking beats to ignore
            -- ectopic beats, typically. Excluding them matters: a run of PVCs
            produces RR irregularity indistinguishable from AF by these
            measures, so classifying beats first is what keeps this from
            reporting AF whenever a subject has frequent ectopy.

            The excluded beats are not deleted from the series. Deleting a
            beat and then differencing merges its two neighbouring intervals
            into one long one, manufacturing exactly the irregularity the
            exclusion was meant to remove. Instead the two intervals touching
            an excluded beat are dropped, and the rest are kept as measured.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < MIN_INTERVALS + 1:
        return []

    times = peaks / float(fs)
    rr = np.diff(times)
    rr_t = times[1:]

    valid = np.ones(rr.size, dtype=bool)
    if exclude is not None:
        bad = np.asarray(exclude, dtype=bool)
        if bad.size == peaks.size:
            # Interval i spans peaks i and i+1; drop it if either is excluded.
            valid &= ~bad[:-1]
            valid &= ~bad[1:]

    end = float(times[-1])
    out = []
    start = 0.0
    while start < end:
        stop = start + window_s
        sel = (rr_t >= start) & (rr_t < stop) & valid
        if np.count_nonzero(sel) >= MIN_INTERVALS:
            out.append(assess_af_window(rr[sel], start_s=start, end_s=stop))
        start = stop
    return out


def af_burden(windows: list) -> float:
    """Fraction of assessed time judged to be AF."""
    if not windows:
        return 0.0
    total = sum(w.end_s - w.start_s for w in windows)
    af = sum(w.end_s - w.start_s for w in windows if w.is_af)
    return af / total if total > 0 else 0.0


def af_episodes(windows: list, min_duration_s: float = 30.0) -> list:
    """Merge adjacent AF windows into episodes.

    Returns ``(start_s, end_s)`` spans. Episodes shorter than
    ``min_duration_s`` are dropped: clinical definitions of AF require the
    rhythm to be sustained, and a single flagged window is more often ectopy
    or artefact than a genuine paroxysm.
    """
    episodes = []
    current = None
    for w in windows:
        if w.is_af:
            if current is None:
                current = [w.start_s, w.end_s]
            else:
                current[1] = w.end_s
        elif current is not None:
            episodes.append(tuple(current))
            current = None
    if current is not None:
        episodes.append(tuple(current))

    return [e for e in episodes if (e[1] - e[0]) >= min_duration_s]
