"""ECG filtering: baseline-wander removal, powerline notch, bandpass.

All filters are zero-phase (``filtfilt``). Phase distortion would shift the
fiducial points we later measure, and any timing error is doubly damaging
here because cross-sensor comparison depends on beats from different sensors
being placed on the same timeline correctly.

Each function has a raw-array form and a ``ChannelRecord`` form, so filters
can be applied to a whole ``SynchronizedRecording`` via ``map_channels``.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from ..io.record import ChannelRecord

# Analysis band. The low edge removes respiration/electrode drift; the high
# edge keeps QRS energy (mostly 8-25 Hz) while dropping EMG noise.
DEFAULT_LOW_HZ = 0.5
DEFAULT_HIGH_HZ = 40.0


def _nyquist_guard(fs: float, hz: float, name: str) -> float:
    """Clamp a cutoff below Nyquist so low-rate records do not blow up."""
    limit = 0.99 * (fs / 2.0)
    if hz >= limit:
        return limit
    if hz <= 0:
        raise ValueError(f"{name} must be positive, got {hz}")
    return hz


def bandpass(
    x: np.ndarray,
    fs: float,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass."""
    x = np.asarray(x, dtype=np.float64)
    low_hz = _nyquist_guard(fs, low_hz, "low_hz")
    high_hz = _nyquist_guard(fs, high_hz, "high_hz")
    if low_hz >= high_hz:
        raise ValueError(f"low_hz ({low_hz}) must be below high_hz ({high_hz})")
    sos = sps.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def highpass(x: np.ndarray, fs: float, cutoff_hz: float = DEFAULT_LOW_HZ,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth highpass, i.e. baseline-wander removal."""
    x = np.asarray(x, dtype=np.float64)
    cutoff_hz = _nyquist_guard(fs, cutoff_hz, "cutoff_hz")
    sos = sps.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def remove_baseline_median(x: np.ndarray, fs: float) -> np.ndarray:
    """Baseline removal by cascaded median filters.

    The classic alternative to a highpass: a 200 ms median filter spans the
    QRS and removes it, a following 600 ms median spans P and T, and what
    survives is the baseline, which is then subtracted. Unlike a highpass it
    does not ring around steep QRS edges, so it is the safer choice when ST
    level matters. It is markedly slower on long records.
    """
    x = np.asarray(x, dtype=np.float64)

    def _odd(n: int) -> int:
        n = max(3, int(n))
        return n + 1 if n % 2 == 0 else n

    stage1 = sps.medfilt(x, _odd(0.2 * fs))
    baseline = sps.medfilt(stage1, _odd(0.6 * fs))
    return x - baseline


def notch(x: np.ndarray, fs: float, freq_hz: float = 60.0, q: float = 30.0,
          harmonics: int = 1) -> np.ndarray:
    """Zero-phase notch at mains frequency and optionally its harmonics.

    Use 60 Hz in North America, 50 Hz in most of the rest of the world.
    Harmonics above Nyquist are skipped rather than raising.
    """
    x = np.asarray(x, dtype=np.float64)
    out = x
    for k in range(1, harmonics + 1):
        f0 = freq_hz * k
        if f0 >= 0.99 * (fs / 2.0):
            break
        b, a = sps.iirnotch(f0, q, fs=fs)
        out = sps.filtfilt(b, a, out)
    return out


def preprocess(
    x: np.ndarray,
    fs: float,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    powerline_hz: float | None = 60.0,
    baseline_method: str = "highpass",
) -> np.ndarray:
    """Standard cleaning chain: baseline removal -> notch -> lowpass band.

    Args:
        baseline_method: ``"highpass"`` (fast) or ``"median"`` (preserves ST).
    """
    x = np.asarray(x, dtype=np.float64)

    if baseline_method == "highpass":
        out = highpass(x, fs, low_hz)
    elif baseline_method == "median":
        out = remove_baseline_median(x, fs)
    else:
        raise ValueError(f"unknown baseline_method {baseline_method!r}")

    if powerline_hz is not None and powerline_hz < 0.99 * (fs / 2.0):
        out = notch(out, fs, powerline_hz)

    high_hz = _nyquist_guard(fs, high_hz, "high_hz")
    sos = sps.butter(4, high_hz, btype="lowpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, out)


def preprocess_channel(
    ch: ChannelRecord,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    powerline_hz: float | None = 60.0,
    baseline_method: str = "highpass",
) -> ChannelRecord:
    """``preprocess`` applied to a channel, preserving all of its metadata."""
    return ch.with_signal(
        preprocess(
            ch.signal,
            ch.fs,
            low_hz=low_hz,
            high_hz=high_hz,
            powerline_hz=powerline_hz,
            baseline_method=baseline_method,
        )
    )


def resample_channel(ch: ChannelRecord, target_fs: float) -> ChannelRecord:
    """Resample a channel to ``target_fs``.

    Needed as soon as sensors of different models are mixed in one recording:
    cross-channel analysis assumes a common sample grid.
    """
    if np.isclose(ch.fs, target_fs):
        return ch
    n_out = int(round(ch.n_samples * target_fs / ch.fs))
    resampled = sps.resample_poly(
        ch.signal,
        up=int(round(target_fs)),
        down=int(round(ch.fs)),
    )
    # resample_poly's length can differ by a sample from the exact ratio.
    if resampled.size != n_out:
        resampled = np.resize(resampled, n_out)
    from dataclasses import replace

    return replace(ch, signal=resampled, fs=float(target_fs))
