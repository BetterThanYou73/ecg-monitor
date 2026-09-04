"""Chunked detection over long recordings.

Detecting on a whole 24-hour array means holding the signal plus four
Pan-Tompkins intermediate arrays in memory at once -- at 360 Hz that is
roughly 1.2 GB in float64, and higher-rate sensors make it worse. Processing
in chunks bounds memory regardless of recording length.

The subtlety is the chunk boundary. A beat sitting on a seam can be missed
by both chunks, or found by both and counted twice. Chunks therefore overlap,
and detections in the overlap are merged rather than concatenated. The
overlap also gives the adaptive thresholds a run-up before the region whose
detections are actually kept, so a chunk does not start cold.
"""

from __future__ import annotations

import numpy as np

from ..io.record import ChannelRecord
from ..preprocessing.filters import preprocess
from ..preprocessing.segments import usable_segments
from .rpeaks import REFRACTORY_S, detect_rpeaks

# Long enough to hold many beats so thresholds settle; short enough to bound
# memory. 300 s at 360 Hz is about 108k samples per intermediate array.
DEFAULT_CHUNK_S = 300.0

# Must exceed the longest plausible RR interval so no beat spans a whole
# overlap, and gives the detector a warm-up run before the kept region.
DEFAULT_OVERLAP_S = 5.0


def detect_rpeaks_chunked(
    x: np.ndarray,
    fs: float,
    chunk_s: float = DEFAULT_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    recover_missed: bool = False,
) -> np.ndarray:
    """Detect R peaks in overlapping chunks, returning global sample indices.

    Results are intended to match whole-signal detection closely; they are not
    guaranteed identical, because the adaptive thresholds see a different
    history in each chunk. Differences concentrate near seams.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    chunk = int(round(chunk_s * fs))
    overlap = int(round(overlap_s * fs))

    if n == 0:
        return np.array([], dtype=int)
    if chunk <= 0 or n <= chunk:
        return detect_rpeaks(x, fs, recover_missed=recover_missed)

    refractory = int(round(REFRACTORY_S * fs))
    step = max(1, chunk - overlap)
    found: list[np.ndarray] = []

    start = 0
    while start < n:
        stop = min(n, start + chunk)
        seg = x[start:stop]
        if seg.size < int(fs):
            break

        peaks = detect_rpeaks(seg, fs, recover_missed=recover_missed) + start

        # Keep only detections owned by this chunk. The first chunk owns from
        # its start; later chunks cede the leading half-overlap to the
        # previous chunk, which already saw that region with warmer thresholds.
        lo = start if start == 0 else start + overlap // 2
        hi = stop if stop >= n else stop - (overlap - overlap // 2)
        found.append(peaks[(peaks >= lo) & (peaks < hi)])

        if stop >= n:
            break
        start += step

    if not found:
        return np.array([], dtype=int)

    merged = np.unique(np.concatenate(found))
    return _dedupe(merged, refractory)


def _dedupe(peaks: np.ndarray, refractory: int) -> np.ndarray:
    """Collapse detections closer together than the refractory period.

    Boundary handling can still leave two detections of one beat when a chunk
    seam falls inside a QRS complex; this removes them without needing the
    signal, keeping the earlier index.
    """
    if peaks.size < 2:
        return peaks
    kept = [int(peaks[0])]
    for p in peaks[1:]:
        if int(p) - kept[-1] >= refractory:
            kept.append(int(p))
    return np.array(kept, dtype=int)


def process_channel_long(
    ch: ChannelRecord,
    chunk_s: float = DEFAULT_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    powerline_hz: float | None = 60.0,
    skip_dropouts: bool = True,
    recover_missed: bool = False,
) -> dict:
    """Filter and detect over a long channel, skipping unusable stretches.

    Returns a dict with ``peaks`` (global sample indices), ``coverage`` and
    the segments analysed. Filtering happens per segment so that a dropout
    cannot ring through the filters into neighbouring good signal.
    """
    from ..preprocessing.segments import coverage as _coverage

    cov = _coverage(ch)

    if skip_dropouts:
        segments = usable_segments(ch)
    else:
        from ..preprocessing.segments import Segment

        segments = [Segment(0, ch.n_samples, ch.fs, True)]

    all_peaks: list[np.ndarray] = []
    for seg in segments:
        raw = ch.signal[seg.start_sample : seg.end_sample]
        if raw.size < int(ch.fs):
            continue
        cleaned = preprocess(raw, ch.fs, powerline_hz=powerline_hz)
        peaks = detect_rpeaks_chunked(
            cleaned, ch.fs, chunk_s=chunk_s, overlap_s=overlap_s,
            recover_missed=recover_missed,
        )
        all_peaks.append(peaks + seg.start_sample)

    peaks = (
        np.unique(np.concatenate(all_peaks)) if all_peaks else np.array([], dtype=int)
    )

    return {
        "peaks": peaks,
        "coverage": cov,
        "segments": segments,
        "n_beats": int(peaks.size),
    }
