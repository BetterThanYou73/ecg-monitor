"""Signal-quality assessment.

Quality scoring matters more for this project than for a clinical 12-lead
system. Sensors sit at different positions, so some will inevitably record
worse than others, and the multi-sensor fusion work depends on being able to
say *which* sensor to trust at a given moment rather than averaging good
signal together with garbage.

Scores are computed per fixed-length window so that quality can vary across
a 24-hour recording, which it certainly will once a subject is moving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from ..io.record import ChannelRecord

DEFAULT_WINDOW_S = 10.0

# Fraction of spectral power expected inside the ECG band for a clean signal.
QRS_BAND = (5.0, 15.0)
ECG_BAND = (0.5, 40.0)


@dataclass
class QualityWindow:
    """Quality metrics for one window of one channel."""

    start_s: float
    end_s: float
    sqi: float           # overall score in [0, 1]
    power_ratio: float   # in-band power / total power
    flatline_frac: float # fraction of samples with near-zero derivative
    saturation_frac: float  # fraction of samples pinned at the extremes
    amplitude_mv: float  # robust peak-to-peak estimate

    @property
    def is_usable(self) -> bool:
        return self.sqi >= 0.5


def _band_power_ratio(x: np.ndarray, fs: float) -> float:
    """Fraction of total power falling inside the ECG band."""
    if x.size < 16:
        return 0.0
    nperseg = min(x.size, max(64, int(round(2 * fs))))
    freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg)
    total = float(np.trapezoid(psd, freqs)) if hasattr(np, "trapezoid") else float(
        np.trapz(psd, freqs)
    )
    if total <= 0:
        return 0.0
    mask = (freqs >= ECG_BAND[0]) & (freqs <= min(ECG_BAND[1], fs / 2))
    if not np.any(mask):
        return 0.0
    inband = float(np.trapezoid(psd[mask], freqs[mask])) if hasattr(np, "trapezoid") else float(
        np.trapz(psd[mask], freqs[mask])
    )
    return float(np.clip(inband / total, 0.0, 1.0))


def _flatline_fraction(x: np.ndarray, fs: float) -> float:
    """Fraction of the window where the signal barely moves.

    A detached electrode produces a flat or near-flat trace, which no
    frequency-domain measure reliably flags.
    """
    if x.size < 2:
        return 1.0
    d = np.abs(np.diff(x))
    scale = np.median(d)
    if scale <= 0:
        return 1.0
    return float(np.mean(d < 0.05 * scale))


def _saturation_fraction(x: np.ndarray) -> float:
    """Fraction of samples sitting at the observed rail values (clipping)."""
    if x.size == 0:
        return 1.0
    lo, hi = float(np.min(x)), float(np.max(x))
    if np.isclose(lo, hi):
        return 1.0
    tol = 1e-6 * (hi - lo)
    return float(np.mean((x <= lo + tol) | (x >= hi - tol)))


def _robust_amplitude(x: np.ndarray) -> float:
    """Peak-to-peak estimate that ignores spikes (99th minus 1st percentile)."""
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 99) - np.percentile(x, 1))


def assess_window(x: np.ndarray, fs: float, start_s: float = 0.0) -> QualityWindow:
    """Score a single window of ECG."""
    x = np.asarray(x, dtype=np.float64)
    power_ratio = _band_power_ratio(x, fs)
    flat = _flatline_fraction(x, fs)
    sat = _saturation_fraction(x)
    amp = _robust_amplitude(x)

    # Combine: in-band power carries the score, flatline and clipping veto it.
    sqi = power_ratio * (1.0 - flat) * (1.0 - min(1.0, sat * 5.0))
    if amp <= 0:
        sqi = 0.0

    return QualityWindow(
        start_s=start_s,
        end_s=start_s + x.size / fs,
        sqi=float(np.clip(sqi, 0.0, 1.0)),
        power_ratio=power_ratio,
        flatline_frac=flat,
        saturation_frac=sat,
        amplitude_mv=amp,
    )


def assess_channel(
    ch: ChannelRecord, window_s: float = DEFAULT_WINDOW_S
) -> list[QualityWindow]:
    """Score a channel window by window across its whole duration."""
    n_win = max(1, int(round(window_s * ch.fs)))
    out: list[QualityWindow] = []
    for start in range(0, max(1, ch.n_samples), n_win):
        seg = ch.signal[start : start + n_win]
        if seg.size < n_win // 2:  # ignore a short trailing remnant
            break
        out.append(assess_window(seg, ch.fs, start_s=start / ch.fs))
    return out


def channel_sqi(ch: ChannelRecord, window_s: float = DEFAULT_WINDOW_S) -> float:
    """Median window SQI, as a single number for the whole channel."""
    windows = assess_channel(ch, window_s=window_s)
    if not windows:
        return 0.0
    return float(np.median([w.sqi for w in windows]))


def rank_channels(recording, window_s: float = DEFAULT_WINDOW_S) -> list[tuple[str, float]]:
    """Rank sensors best-quality-first.

    This is the routine the multi-sensor work will lean on: given several
    positions recording the same heart, decide which to believe.
    """
    scored = [(ch.sensor_id, channel_sqi(ch, window_s=window_s)) for ch in recording]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
