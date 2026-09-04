"""Recording-gap and dropout handling for long recordings.

A 24-hour ambulatory recording is not 24 hours of usable ECG. Electrodes
detach, subjects move, batteries interrupt, and wireless sensors drop
packets. Analysing straight through those stretches produces confident
nonsense: a flatline yields no beats, which reads downstream as a long pause,
which reads as an arrhythmia.

The approach here is to find the unusable stretches first, analyse only the
usable ones, and report coverage alongside every result, so a burden figure
is always relative to how much signal actually existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.record import ChannelRecord

# A dropout has to persist to count; brief dips are handled by the filters.
MIN_DROPOUT_S = 1.0


@dataclass
class Segment:
    """A contiguous stretch of one channel, marked usable or not."""

    start_sample: int
    end_sample: int  # exclusive
    fs: float
    usable: bool
    reason: str = ""

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def start_s(self) -> float:
        return self.start_sample / self.fs

    @property
    def end_s(self) -> float:
        return self.end_sample / self.fs

    def __repr__(self) -> str:
        state = "usable" if self.usable else f"unusable({self.reason})"
        return f"Segment({self.start_s:.1f}-{self.end_s:.1f}s, {state})"


@dataclass
class Coverage:
    """How much of a recording was actually analysable."""

    total_s: float
    usable_s: float
    n_gaps: int

    @property
    def fraction(self) -> float:
        return self.usable_s / self.total_s if self.total_s > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.usable_s / 3600:.2f} h usable of {self.total_s / 3600:.2f} h "
            f"({self.fraction * 100:.1f}%), {self.n_gaps} gap(s)"
        )


def _flat_mask(x: np.ndarray, fs: float, window_s: float = 0.5) -> np.ndarray:
    """Boolean mask, per sample, of stretches where the signal barely moves.

    Computed on a coarse grid and then expanded, so cost stays linear on a
    24-hour record rather than quadratic in a rolling comparison.
    """
    n_win = max(1, int(round(window_s * fs)))
    n_blocks = int(np.ceil(x.size / n_win))
    mask = np.zeros(x.size, dtype=bool)

    d = np.abs(np.diff(x, prepend=x[0]))

    # The scale must ignore non-finite samples. A single NaN anywhere makes
    # np.median return NaN, every comparison against it is False, and no
    # flatline is ever detected -- so a data gap would silently disable
    # dropout detection for the whole recording.
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.ones(x.size, dtype=bool)
    scale = float(np.median(finite))
    if scale <= 0:
        return np.ones(x.size, dtype=bool)

    threshold = 0.02 * scale
    for b in range(n_blocks):
        lo = b * n_win
        hi = min(x.size, lo + n_win)
        block = d[lo:hi]
        if block.size == 0:
            continue
        # Non-finite samples are handled by _nan_mask; they must not count as
        # "moving" here, or a gap would mask the flatline surrounding it.
        quiet = np.isfinite(block) & (block < threshold)
        if np.mean(quiet | ~np.isfinite(block)) > 0.9:
            mask[lo:hi] = True
    return mask


def _nan_mask(x: np.ndarray) -> np.ndarray:
    return ~np.isfinite(x)


def find_segments(
    ch: ChannelRecord, min_dropout_s: float = MIN_DROPOUT_S
) -> list[Segment]:
    """Split a channel into alternating usable and unusable segments.

    Unusable means non-finite samples (a true data gap) or a sustained
    flatline (a detached or disconnected electrode).
    """
    x = ch.signal
    if x.size == 0:
        return []

    bad = _nan_mask(x) | _flat_mask(x, ch.fs)

    # Drop dropouts shorter than the minimum: they are noise, not disconnection.
    min_len = max(1, int(round(min_dropout_s * ch.fs)))
    segments: list[Segment] = []
    edges = np.flatnonzero(np.diff(bad.astype(np.int8)) != 0) + 1
    bounds = np.concatenate([[0], edges, [x.size]])

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        lo, hi = int(lo), int(hi)
        is_bad = bool(bad[lo])
        if is_bad and (hi - lo) < min_len:
            is_bad = False  # too short to count as a dropout
        reason = ""
        if is_bad:
            reason = "nan" if not np.all(np.isfinite(x[lo:hi])) else "flatline"
        segments.append(
            Segment(lo, hi, ch.fs, usable=not is_bad, reason=reason)
        )

    # Merge neighbours that ended up with the same verdict.
    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1].usable == seg.usable:
            prev = merged[-1]
            merged[-1] = Segment(
                prev.start_sample, seg.end_sample, ch.fs, prev.usable, prev.reason
            )
        else:
            merged.append(seg)
    return merged


def usable_segments(
    ch: ChannelRecord, min_duration_s: float = 10.0, min_dropout_s: float = MIN_DROPOUT_S
) -> list[Segment]:
    """Usable segments long enough to analyse.

    Very short usable slivers between dropouts are excluded: a detector needs
    several beats before its adaptive thresholds mean anything.
    """
    return [
        s
        for s in find_segments(ch, min_dropout_s=min_dropout_s)
        if s.usable and s.duration_s >= min_duration_s
    ]


def coverage(ch: ChannelRecord, min_dropout_s: float = MIN_DROPOUT_S) -> Coverage:
    """Report how much of the channel is analysable."""
    segs = find_segments(ch, min_dropout_s=min_dropout_s)
    usable = sum(s.duration_s for s in segs if s.usable)
    gaps = sum(1 for s in segs if not s.usable)
    return Coverage(total_s=ch.duration_s, usable_s=usable, n_gaps=gaps)
