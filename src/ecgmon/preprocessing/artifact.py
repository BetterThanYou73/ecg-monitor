"""Motion-artefact detection.

The signal-quality measures in ``quality`` catch gross failures: a dead
electrode, a clipped amplifier, a burst of broadband noise. They do not catch
motion artefact, because motion puts its energy *inside* the ECG band. A
subject rolling over produces a deflection with the amplitude and bandwidth
of a QRS complex, and a band-power ratio sees nothing wrong with it.

That matters here specifically: an artefact that looks like a beat gets
detected as one, and a beat that looks unlike the patient's normal template
gets classified as ectopic. Motion artefact therefore manufactures false
ventricular beats, and is the most likely source of the classifier's
remaining false positives.

Three measures, chosen because they fail in different ways:

* **Kurtosis.** A clean ECG is spiky -- long flat stretches punctuated by
  QRS complexes -- so its amplitude distribution is heavy-tailed. Noise is
  closer to Gaussian. Kurtosis separates the two without reference to
  frequency content.
* **Beat-detector agreement (bSQI).** Two independent detectors agree on real
  beats and disagree on artefact, because each responds to a different aspect
  of the waveform. This needs no threshold tuned to a particular recording,
  which is what makes it transfer between subjects and sensors.
* **Amplitude instability.** Motion shifts the envelope of the signal.
  A sudden change in local amplitude is visible even where the spectrum and
  the beat detections look ordinary.

None is sufficient alone. Kurtosis is fooled by a single large spike, bSQI by
artefact regular enough that both detectors latch onto it, instability by
genuine physiological amplitude changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..io.record import ChannelRecord

DEFAULT_WINDOW_S = 10.0

# A clean ECG window typically has kurtosis well above this; Gaussian noise
# sits at 3. The boundary is deliberately generous -- this is one vote of
# three, not a verdict.
KURTOSIS_CLEAN = 5.0

# Detections within this distance count as the two detectors agreeing.
AGREEMENT_TOLERANCE_S = 0.15


@dataclass
class ArtifactWindow:
    """Artefact assessment for one window."""

    start_s: float
    end_s: float
    kurtosis: float
    bsqi: float              # 0-1, agreement between two beat detectors
    instability: float       # 0-1, relative change in local amplitude
    score: float             # 0-1, combined; higher means more artefact

    @property
    def is_artifact(self) -> bool:
        return self.score >= 0.5

    def __repr__(self) -> str:
        return (
            f"ArtifactWindow({self.start_s:.0f}-{self.end_s:.0f}s, "
            f"score={self.score:.2f}, k={self.kurtosis:.1f}, bSQI={self.bsqi:.2f})"
        )


def kurtosis_score(x: np.ndarray) -> float:
    """Fisher kurtosis of the window (0 for a Gaussian)."""
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size < 8 or np.allclose(finite, finite[0]):
        return 0.0
    return float(stats.kurtosis(finite, fisher=True, bias=False))


def bsqi(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    fs: float,
    tolerance_s: float = AGREEMENT_TOLERANCE_S,
) -> float:
    """Fraction of detections two detectors agree on.

    Defined as matched / (total distinct beats proposed by either), so a
    detector firing on artefact that the other ignores drives the score down
    from both directions.

    Returns 1.0 when neither detector found anything: an empty window is not
    evidence of artefact, and flatline detection is handled elsewhere.
    """
    a = np.sort(np.asarray(peaks_a, dtype=np.int64))
    b = np.sort(np.asarray(peaks_b, dtype=np.int64))
    if a.size == 0 and b.size == 0:
        return 1.0
    if a.size == 0 or b.size == 0:
        return 0.0

    tol = tolerance_s * fs
    matched = 0
    j = 0
    used = np.zeros(b.size, dtype=bool)
    for x in a:
        while j < b.size and b[j] < x - tol:
            j += 1
        k = j
        best, best_d = -1, None
        while k < b.size and b[k] <= x + tol:
            if not used[k]:
                d = abs(int(b[k]) - int(x))
                if best_d is None or d < best_d:
                    best, best_d = k, d
            k += 1
        if best >= 0:
            used[best] = True
            matched += 1

    union = a.size + b.size - matched
    return float(matched / union) if union else 1.0


def amplitude_instability(x: np.ndarray, fs: float, sub_window_s: float = 1.0) -> float:
    """Relative variation of the local amplitude envelope, clipped to [0, 1].

    Computed as the median absolute deviation of per-second peak-to-peak
    amplitude, divided by its median. Using medians rather than means keeps a
    single large motion spike from defining the baseline it is measured
    against.
    """
    x = np.asarray(x, dtype=np.float64)
    n = max(1, int(round(sub_window_s * fs)))
    if x.size < 2 * n:
        return 0.0

    n_sub = x.size // n
    amps = np.empty(n_sub)
    for i in range(n_sub):
        seg = x[i * n : (i + 1) * n]
        finite = seg[np.isfinite(seg)]
        amps[i] = (
            np.percentile(finite, 97.5) - np.percentile(finite, 2.5)
            if finite.size > 4
            else 0.0
        )

    med = np.median(amps)
    if med <= 0:
        return 1.0
    mad = np.median(np.abs(amps - med))
    return float(np.clip(mad / med, 0.0, 1.0))


def assess_artifact_window(
    x: np.ndarray,
    fs: float,
    peaks_a: np.ndarray | None = None,
    peaks_b: np.ndarray | None = None,
    start_s: float = 0.0,
) -> ArtifactWindow:
    """Score one window for motion artefact.

    ``peaks_a`` / ``peaks_b`` are detections from two independent detectors,
    in window-local sample indices. When omitted, bSQI is not counted and the
    score rests on the other two measures.
    """
    x = np.asarray(x, dtype=np.float64)

    k = kurtosis_score(x)
    inst = amplitude_instability(x, fs)

    have_bsqi = peaks_a is not None and peaks_b is not None
    b = bsqi(peaks_a, peaks_b, fs) if have_bsqi else 1.0

    # Each measure becomes a 0-1 "badness". Kurtosis below the clean threshold
    # is suspicious; above it, not.
    bad_kurt = float(np.clip(1.0 - (k / KURTOSIS_CLEAN), 0.0, 1.0))
    bad_bsqi = float(np.clip(1.0 - b, 0.0, 1.0))
    bad_inst = float(np.clip(inst / 0.5, 0.0, 1.0))

    if have_bsqi:
        # Detector disagreement is weighted highest: it is the measure that
        # does not depend on a threshold calibrated to a particular signal.
        score = 0.45 * bad_bsqi + 0.30 * bad_kurt + 0.25 * bad_inst
    else:
        score = 0.55 * bad_kurt + 0.45 * bad_inst

    return ArtifactWindow(
        start_s=start_s,
        end_s=start_s + x.size / fs,
        kurtosis=k,
        bsqi=b,
        instability=inst,
        score=float(np.clip(score, 0.0, 1.0)),
    )


def assess_artifact(
    ch: ChannelRecord,
    window_s: float = DEFAULT_WINDOW_S,
    use_bsqi: bool = True,
) -> list:
    """Score a channel for motion artefact, window by window.

    Both detectors run once over the whole channel rather than per window, so
    their adaptive thresholds see the full recording instead of restarting
    cold every ten seconds.
    """
    peaks_a = peaks_b = None
    if use_bsqi:
        from ..detection.rpeaks import detect_rpeaks, detect_rpeaks_neurokit

        try:
            peaks_a = detect_rpeaks(ch.signal, ch.fs)
            peaks_b = detect_rpeaks_neurokit(ch.signal, ch.fs)
        except Exception:
            # A detector failing on a pathological signal is itself weak
            # evidence of artefact, but not worth aborting the whole channel.
            peaks_a = peaks_b = None

    n_win = max(1, int(round(window_s * ch.fs)))
    out = []
    for start in range(0, max(1, ch.n_samples), n_win):
        seg = ch.signal[start : start + n_win]
        if seg.size < n_win // 2:
            break
        if peaks_a is not None and peaks_b is not None:
            stop = start + seg.size
            a = peaks_a[(peaks_a >= start) & (peaks_a < stop)] - start
            b = peaks_b[(peaks_b >= start) & (peaks_b < stop)] - start
        else:
            a = b = None
        out.append(
            assess_artifact_window(seg, ch.fs, a, b, start_s=start / ch.fs)
        )
    return out


def artifact_mask(
    ch: ChannelRecord, window_s: float = DEFAULT_WINDOW_S, threshold: float = 0.5
) -> np.ndarray:
    """Per-sample boolean mask of windows judged to be artefact."""
    windows = assess_artifact(ch, window_s=window_s)
    mask = np.zeros(ch.n_samples, dtype=bool)
    n_win = max(1, int(round(window_s * ch.fs)))
    for i, w in enumerate(windows):
        if w.score >= threshold:
            mask[i * n_win : (i + 1) * n_win] = True
    return mask


def beats_in_artifact(
    peaks: np.ndarray,
    ch: ChannelRecord,
    window_s: float = DEFAULT_WINDOW_S,
    threshold: float = 0.5,
) -> np.ndarray:
    """Boolean mask over ``peaks``: True where the beat falls in artefact.

    This is the form the classifier needs -- not "which samples are bad" but
    "which detected beats should be distrusted".
    """
    mask = artifact_mask(ch, window_s=window_s, threshold=threshold)
    peaks = np.asarray(peaks, dtype=np.int64)
    inside = (peaks >= 0) & (peaks < mask.size)
    out = np.zeros(peaks.size, dtype=bool)
    out[inside] = mask[peaks[inside]]
    return out
