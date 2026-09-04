"""R-peak / QRS detection.

The primary detector is Pan-Tompkins, implemented here rather than called
from a library so that its intermediate stages stay inspectable: when a
sensor position gives poor signal quality, being able to look at the
integrated waveform and the adaptive thresholds is what tells you whether
the problem is the electrode or the detector.

``detect_rpeaks_neurokit`` wraps NeuroKit2 as a cross-check. Agreement
between two independent detectors is a useful confidence signal on real
sensor data, where there are no reference annotations to compare against.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..io.record import ChannelRecord

# Physiological limits used to constrain the search.
REFRACTORY_S = 0.20  # no two R peaks closer than this (300 bpm ceiling)
T_WAVE_WINDOW_S = 0.36  # candidates this close are checked for T-wave slope


def _integration_window(fs: float) -> int:
    """Moving-window integrator width: ~150 ms, the width of a wide QRS."""
    return max(1, int(round(0.150 * fs)))


def pan_tompkins_stages(x: np.ndarray, fs: float) -> dict:
    """Run the Pan-Tompkins preprocessing chain and return every stage.

    Returns a dict with ``bandpassed``, ``differentiated``, ``squared`` and
    ``integrated`` arrays. Useful for debugging and for the diagnostic plot.
    """
    x = np.asarray(x, dtype=np.float64)

    # 5-15 Hz passband isolates QRS energy from P/T waves and from EMG.
    low = max(0.5, min(5.0, 0.99 * fs / 2 - 1e-6))
    high = min(15.0, 0.99 * fs / 2)
    if low >= high:
        bandpassed = x - np.mean(x)
    else:
        sos = sps.butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
        bandpassed = sps.sosfiltfilt(sos, x)

    # Derivative emphasises the steep QRS slope over slower deflections.
    differentiated = np.gradient(bandpassed) * fs
    squared = differentiated ** 2

    win = _integration_window(fs)
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    return {
        "bandpassed": bandpassed,
        "differentiated": differentiated,
        "squared": squared,
        "integrated": integrated,
    }


def _local_maxima(x: np.ndarray, min_distance: int) -> np.ndarray:
    peaks, _ = sps.find_peaks(x, distance=max(1, min_distance))
    return peaks


def detect_rpeaks(
    x: np.ndarray,
    fs: float,
    return_stages: bool = False,
    recover_missed: bool = False,
):
    """Detect R-peak sample indices with Pan-Tompkins adaptive thresholding.

    The detector maintains running estimates of signal and noise peak
    amplitudes, adapts its threshold to their difference, and applies a
    search-back: if no peak is found within 1.66x the recent RR interval, the
    window is re-scanned at half threshold to recover a missed beat.

    Args:
        x: Raw or filtered single-channel ECG.
        fs: Sampling rate in Hz.
        return_stages: Also return the intermediate waveforms.
        recover_missed: Run ``recover_missed_beats`` as a second pass. This
            raises sensitivity and lowers PPV -- see that function for the
            measured trade-off. Off by default because inventing beats inside
            a genuine pause corrupts exactly the rhythm analysis that pause
            feeds into.

    Returns:
        Array of sample indices, or ``(indices, stages_dict)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < int(fs):  # under a second of data: nothing meaningful to do
        empty = np.array([], dtype=int)
        return (empty, pan_tompkins_stages(x, fs)) if return_stages else empty

    stages = pan_tompkins_stages(x, fs)
    integrated = stages["integrated"]

    refractory = int(round(REFRACTORY_S * fs))
    candidates = _local_maxima(integrated, refractory)
    if candidates.size == 0:
        empty = np.array([], dtype=int)
        return (empty, stages) if return_stages else empty

    # Initialise thresholds from the first two seconds.
    init = integrated[: int(round(2 * fs))]
    spk = float(np.max(init)) * 0.25 if init.size else float(np.max(integrated)) * 0.25
    npk = float(np.mean(init)) * 0.5 if init.size else 0.0
    threshold = npk + 0.25 * (spk - npk)

    signal_peaks: list[int] = []
    rr_recent: list[float] = []
    last_peak = -refractory

    def _rr_missed_limit() -> float | None:
        if len(rr_recent) < 2:
            return None
        return 1.66 * float(np.mean(rr_recent[-8:]))

    for idx in candidates:
        amp = integrated[idx]

        # Search-back: a suspiciously long gap suggests a beat was missed.
        limit = _rr_missed_limit()
        if limit is not None and signal_peaks:
            gap = (idx - last_peak) / fs
            if gap > limit:
                lo, hi = last_peak + refractory, idx
                if hi > lo:
                    window = integrated[lo:hi]
                    if window.size:
                        rel = int(np.argmax(window))
                        recovered = lo + rel
                        if integrated[recovered] > 0.5 * threshold:
                            if signal_peaks:
                                rr_recent.append((recovered - last_peak) / fs)
                            signal_peaks.append(recovered)
                            last_peak = recovered
                            spk = 0.25 * integrated[recovered] + 0.75 * spk

        if amp > threshold and (idx - last_peak) >= refractory:
            if signal_peaks:
                rr_recent.append((idx - last_peak) / fs)
            signal_peaks.append(idx)
            last_peak = idx
            spk = 0.125 * amp + 0.875 * spk
        else:
            npk = 0.125 * amp + 0.875 * npk

        threshold = npk + 0.25 * (spk - npk)

    # The integrator delays and broadens the peak; relocate each detection to
    # the true R extremum in the bandpassed signal.
    refined = _refine_to_extremum(stages["bandpassed"], np.array(signal_peaks, dtype=int), fs)
    refined = _enforce_refractory(refined, stages["bandpassed"], refractory)

    if recover_missed:
        refined = recover_missed_beats(refined, stages, fs)

    return (refined, stages) if return_stages else refined


def recover_missed_beats(
    peaks: np.ndarray,
    stages: dict,
    fs: float,
    frac: float = 0.35,
    max_ratio: float = 1.5,
) -> np.ndarray:
    """Second pass: re-scan abnormally long gaps at a locally-derived threshold.

    The adaptive threshold is driven by a single running signal-peak estimate,
    so a record mixing large ectopic beats with small normal ones pulls the
    threshold above the small beats and loses them in runs. This pass looks
    only inside gaps longer than ``max_ratio`` times the median RR interval
    and thresholds against *that gap's own* amplitude distribution, which is
    what makes low-amplitude beats recoverable.

    Measured on the eight-record MIT-BIH benchmark set:

    ==============  ==============  ==============
    channel         without         with
    ==============  ==============  ==============
    lead 0          98.37 / 99.27   99.50 / 97.96
    lead 1          83.96 / 97.27   89.36 / 94.89
    ==============  ==============  ==============

    (sensitivity / PPV, percent.) Sensitivity rises and PPV falls, because a
    long gap is not always a missed beat -- it is sometimes a real pause, and
    this pass will populate it regardless. Prefer it when a downstream stage
    tolerates false beats better than missing ones.
    """
    from scipy import signal as _sps

    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 3:
        return peaks

    integrated = stages["integrated"]
    refractory = int(round(REFRACTORY_S * fs))
    median_rr = float(np.median(np.diff(peaks)))
    if median_rr <= 0:
        return peaks

    extra: list[int] = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        gap = int(b) - int(a)
        if gap <= max_ratio * median_rr:
            continue
        lo, hi = int(a) + refractory, int(b) - refractory
        if hi <= lo:
            continue
        seg = integrated[lo:hi]
        if seg.size < 3:
            continue

        expected = int(round(gap / median_rr)) - 1
        if expected < 1:
            continue

        cands, _ = _sps.find_peaks(
            seg, distance=refractory, height=frac * np.percentile(seg, 95)
        )
        if cands.size == 0:
            continue
        strongest = np.argsort(seg[cands])[::-1][:expected]
        extra.extend((lo + cands[strongest]).tolist())

    if not extra:
        return peaks

    merged = np.unique(np.concatenate([peaks, np.array(extra, dtype=int)]))
    merged = _refine_to_extremum(stages["bandpassed"], merged, fs)
    return _enforce_refractory(merged, stages["bandpassed"], refractory)


def _refine_to_extremum(x: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    """Snap each detection to the largest-magnitude sample nearby.

    Magnitude rather than value, because R waves are negative in some sensor
    positions and we do not want placement to depend on lead polarity.
    """
    if peaks.size == 0:
        return peaks
    half = max(1, int(round(0.05 * fs)))
    out = []
    for p in peaks:
        lo = max(0, p - half)
        hi = min(x.size, p + half + 1)
        if hi <= lo:
            out.append(p)
            continue
        out.append(lo + int(np.argmax(np.abs(x[lo:hi]))))
    return np.unique(np.array(out, dtype=int))


def _enforce_refractory(peaks: np.ndarray, x: np.ndarray, refractory: int) -> np.ndarray:
    """Drop the weaker of any two detections closer together than refractory."""
    if peaks.size < 2:
        return peaks
    kept = [int(peaks[0])]
    for p in peaks[1:]:
        p = int(p)
        if p - kept[-1] < refractory:
            if abs(x[p]) > abs(x[kept[-1]]):
                kept[-1] = p
        else:
            kept.append(p)
    return np.array(kept, dtype=int)


def detect_rpeaks_neurokit(x: np.ndarray, fs: float) -> np.ndarray:
    """Second opinion via NeuroKit2, for agreement checks on unlabelled data."""
    import neurokit2 as nk

    cleaned = nk.ecg_clean(np.asarray(x, dtype=np.float64), sampling_rate=int(fs))
    _, info = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
    return np.asarray(info["ECG_R_Peaks"], dtype=int)


def detect_channel(
    ch: ChannelRecord, method: str = "pan_tompkins", recover_missed: bool = False
) -> np.ndarray:
    """Detect R peaks on a ``ChannelRecord``."""
    if method == "pan_tompkins":
        return detect_rpeaks(ch.signal, ch.fs, recover_missed=recover_missed)
    if method == "neurokit":
        return detect_rpeaks_neurokit(ch.signal, ch.fs)
    raise ValueError(f"unknown method {method!r}")


def detect_recording(
    recording, method: str = "pan_tompkins", recover_missed: bool = False
) -> dict:
    """Detect R peaks on every channel.

    Returns ``{sensor_id: indices}``. Indices stay in each channel's own
    sample space; convert with ``ChannelRecord.elapsed_since`` before
    comparing detections across sensors.
    """
    return {
        ch.sensor_id: detect_channel(ch, method=method, recover_missed=recover_missed)
        for ch in recording
    }
