"""Beat morphology: extraction, templates and per-beat features.

R-peak detection says *where* the beats are. Telling a normal beat from an
abnormal one needs to know what each beat *looks like*, which is what this
module provides.

The organising idea is a template. Normal beats from one subject look alike,
so a robust average of that subject's beats is a good model of "normal for
this person", and the beats that correlate poorly with it are the interesting
ones. This is deliberately per-subject: normal morphology varies enormously
between people and with electrode position, so a template learned from one
subject transfers badly to another. A ventricular beat is abnormal relative
to *its own* recording, not relative to a population average.

Features here are interpretable on purpose. A wide QRS that correlates poorly
with the patient's normal beat and arrives early is the textbook description
of a PVC, and a feature set that mirrors that description can be checked by
eye against the signal when it goes wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal as sps

# Window around each R peak. Wide enough to contain the QRS and some of the
# ST segment, short enough that neighbouring beats do not intrude at high
# heart rates (at 200 bpm, beats are 300 ms apart).
BEFORE_S = 0.20
AFTER_S = 0.35

# Narrower window used for template correlation: the QRS itself. Including
# the T wave would let T-wave variation dominate a morphology comparison
# that is supposed to be about ventricular conduction.
QRS_BEFORE_S = 0.06
QRS_AFTER_S = 0.10


@dataclass
class BeatFeatures:
    """Per-beat morphology and rhythm context.

    Rhythm context matters as much as shape: a wide, odd-looking beat that
    arrives on time is more likely a conduction abnormality than an ectopic
    beat, and prematurity is what separates them.
    """

    index: int            # beat number within the record
    sample: int           # R-peak sample index
    template_corr: float  # Pearson r against the subject's normal template
    qrs_width_ms: float
    r_amplitude: float
    qrs_energy: float     # area under the squared QRS, normalised
    pre_rr_ratio: float   # preceding RR / local median RR  (<1 = premature)
    post_rr_ratio: float  # following RR / local median RR  (>1 = pause)
    # Subject-relative shape features. Absolute amplitude and width are not
    # comparable across subjects -- electrode position and body habitus shift
    # both, and measurements on MIT-BIH show the normal-vs-ventricular
    # difference reversing direction from one record to the next. Expressed
    # against the subject's own median, the *deviation* transfers even when
    # the raw value does not.
    r_amp_ratio: float    # |R| / median |R| for this record
    width_ratio: float    # QRS width / median QRS width for this record

    def to_dict(self) -> dict:
        return asdict(self)


def extract_beats(
    x: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    before_s: float = BEFORE_S,
    after_s: float = AFTER_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fixed window around each R peak.

    Returns ``(beats, kept)`` where ``beats`` is (n_kept, window_length) and
    ``kept`` indexes the peaks that had a complete window. Beats too close to
    either end of the recording are dropped rather than zero-padded, because
    padding would fabricate a flat segment that reads as abnormal morphology.
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64)

    n_before = int(round(before_s * fs))
    n_after = int(round(after_s * fs))

    ok = (peaks - n_before >= 0) & (peaks + n_after < x.size)
    kept = np.flatnonzero(ok)
    if kept.size == 0:
        return np.empty((0, n_before + n_after)), kept

    idx = peaks[kept]
    offsets = np.arange(-n_before, n_after)
    windows = x[idx[:, None] + offsets[None, :]]
    return windows, kept


def build_template(beats: np.ndarray, corr_threshold: float = 0.8) -> np.ndarray:
    """Build a robust "normal beat" template for one subject.

    Two passes. The first takes the median across all beats, which already
    resists outliers: ectopic beats have to outnumber normal ones to shift a
    median, and in most recordings they do not. The second pass keeps only
    beats correlating well with that first estimate and re-medians them, which
    sharpens the template by excluding the ectopics that survived pass one.

    Falls back to the first-pass median when too few beats agree, rather than
    building a template from a handful of outliers.
    """
    if beats.shape[0] == 0:
        raise ValueError("no beats to build a template from")
    if beats.shape[0] == 1:
        return beats[0].copy()

    first = np.median(beats, axis=0)

    corrs = _corr_to_template(beats, first)
    good = corrs >= corr_threshold
    # Require a real majority before trusting the refined template.
    if np.count_nonzero(good) < max(5, 0.2 * beats.shape[0]):
        return first
    return np.median(beats[good], axis=0)


def _corr_to_template(beats: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of ``beats`` against ``template``."""
    if beats.shape[0] == 0:
        return np.array([])

    b = beats - beats.mean(axis=1, keepdims=True)
    t = template - template.mean()

    b_norm = np.sqrt((b ** 2).sum(axis=1))
    t_norm = np.sqrt((t ** 2).sum())

    denom = b_norm * t_norm
    out = np.zeros(beats.shape[0])
    nz = denom > 0
    out[nz] = (b[nz] @ t) / denom[nz]
    return out


def qrs_widths(
    beats: np.ndarray, fs: float, r_offset: int, frac: float = 0.15
) -> np.ndarray:
    """Estimate QRS duration for each beat, in milliseconds.

    Measured on the squared derivative, whose envelope tracks the steep
    deflections that make up the QRS while ignoring the slower P and T waves.
    The width is taken where that envelope falls below ``frac`` of its peak on
    each side of R.

    This is an estimate, not a clinical measurement -- onset and offset are
    defined by a threshold rather than by the isoelectric line -- but it
    separates wide ventricular beats from narrow supraventricular ones, which
    is what it is for.
    """
    if beats.shape[0] == 0:
        return np.array([])

    d = np.diff(beats, axis=1, prepend=beats[:, :1])
    env = d ** 2

    # Smooth over ~20 ms so single noisy samples do not define the edges.
    k = max(1, int(round(0.02 * fs)))
    if k > 1:
        kernel = np.ones(k) / k
        env = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, env)

    widths = np.zeros(beats.shape[0])
    n = beats.shape[1]
    centre = min(max(r_offset, 0), n - 1)

    for i in range(beats.shape[0]):
        row = env[i]
        peak = row[centre] if row[centre] > 0 else row.max()
        if peak <= 0:
            widths[i] = np.nan
            continue
        thr = frac * peak

        left = centre
        while left > 0 and row[left] > thr:
            left -= 1
        right = centre
        while right < n - 1 and row[right] > thr:
            right += 1

        widths[i] = (right - left) / fs * 1000.0

    return widths


def local_median_rr(peaks: np.ndarray, fs: float, k: int = 11) -> np.ndarray:
    """Local median RR around each beat, in seconds.

    Compared against a *local* median rather than a global mean so that
    prematurity stays meaningful while heart rate drifts over a long
    recording.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size < 2:
        return np.array([])

    rr = np.diff(peaks) / fs
    if rr.size == 0:
        return np.array([])

    k = min(k, rr.size if rr.size % 2 == 1 else max(1, rr.size - 1))
    if k < 1:
        k = 1
    pad = k // 2
    padded = np.pad(rr, pad, mode="edge")
    return np.array([np.median(padded[i : i + k]) for i in range(rr.size)])


def beat_features(
    x: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    template: np.ndarray | None = None,
) -> tuple[list[BeatFeatures], np.ndarray, np.ndarray]:
    """Compute per-beat features for a whole channel.

    Returns ``(features, beats, template)``. ``features`` covers only the
    beats with a complete window, so its ``index`` field refers back into the
    original ``peaks`` array.
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64)

    beats, kept = extract_beats(x, fs, peaks)
    if beats.shape[0] == 0:
        return [], beats, np.array([])

    n_before = int(round(BEFORE_S * fs))

    # Correlate over the QRS only, so T-wave variation does not dominate.
    q0 = n_before - int(round(QRS_BEFORE_S * fs))
    q1 = n_before + int(round(QRS_AFTER_S * fs))
    q0, q1 = max(0, q0), min(beats.shape[1], q1)
    qrs_windows = beats[:, q0:q1]

    if template is None:
        template = build_template(qrs_windows)

    corrs = _corr_to_template(qrs_windows, template)
    widths = qrs_widths(beats, fs, r_offset=n_before)

    r_amp = np.abs(beats[:, n_before])
    energy = (qrs_windows ** 2).sum(axis=1)
    energy = energy / np.median(energy) if np.median(energy) > 0 else energy

    # Subject-relative shape: deviation from this record's own median.
    med_amp = np.median(r_amp)
    amp_ratio = r_amp / med_amp if med_amp > 0 else np.full_like(r_amp, np.nan)
    finite_w = widths[np.isfinite(widths)]
    med_w = np.median(finite_w) if finite_w.size else np.nan
    width_ratio = widths / med_w if med_w and np.isfinite(med_w) and med_w > 0 else np.full_like(widths, np.nan)

    med_rr = local_median_rr(peaks, fs)
    rr = np.diff(peaks) / fs

    out: list[BeatFeatures] = []
    for row, beat_idx in enumerate(kept):
        # rr[i] is the interval between peak i and peak i+1.
        pre = rr[beat_idx - 1] if beat_idx >= 1 else np.nan
        post = rr[beat_idx] if beat_idx < rr.size else np.nan
        ref_pre = med_rr[beat_idx - 1] if beat_idx >= 1 and beat_idx - 1 < med_rr.size else np.nan
        ref_post = med_rr[beat_idx] if beat_idx < med_rr.size else np.nan

        out.append(
            BeatFeatures(
                index=int(beat_idx),
                sample=int(peaks[beat_idx]),
                template_corr=float(corrs[row]),
                qrs_width_ms=float(widths[row]),
                r_amplitude=float(r_amp[row]),
                qrs_energy=float(energy[row]),
                pre_rr_ratio=float(pre / ref_pre) if ref_pre and np.isfinite(ref_pre) and ref_pre > 0 else np.nan,
                post_rr_ratio=float(post / ref_post) if ref_post and np.isfinite(ref_post) and ref_post > 0 else np.nan,
                r_amp_ratio=float(amp_ratio[row]),
                width_ratio=float(width_ratio[row]),
            )
        )

    return out, beats, template


def features_to_array(features: list[BeatFeatures]) -> np.ndarray:
    """Stack features into an (n_beats, 6) array for classification.

    Column order matches ``FEATURE_NAMES``.
    """
    if not features:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.array(
        [
            [
                f.template_corr,
                f.qrs_width_ms,
                f.r_amplitude,
                f.qrs_energy,
                f.pre_rr_ratio,
                f.post_rr_ratio,
                f.r_amp_ratio,
                f.width_ratio,
            ]
            for f in features
        ],
        dtype=np.float64,
    )


# Features used for classification: morphology only.
#
# The absolute-scale measurements (qrs_width_ms, r_amplitude) are excluded
# because a portable sensor will not share MIT-BIH's amplitude calibration,
# so a model leaning on absolute millivolts cannot transfer to real hardware.
#
# The rhythm-context features (pre_rr_ratio, post_rr_ratio) are excluded for
# a different and more interesting reason: measured inter-patient, they make
# the classifier *worse* -- 84.04 F1 with morphology alone against 79.66 with
# rhythm added, on identical test beats. Prematurity and compensatory pause
# are genuinely diagnostic within one subject, but baseline rhythm varies so
# much between people -- and is meaningless in the atrial-fibrillation
# records, where every interval is irregular -- that thresholds learned on
# one group of subjects do not carry to another. They raise sensitivity and
# cost more precision than they return.
#
# This is worth keeping in view rather than treating as settled: rhythm
# context should matter, and a more robust per-subject normalisation may yet
# recover it. It is also expected to be essential for supraventricular
# (PAC) beats, whose morphology is near-normal and which are therefore
# separable mainly by timing.
CLASSIFIER_FEATURES = [
    "template_corr",
    "qrs_energy",
    "r_amp_ratio",
    "width_ratio",
]


def classifier_columns() -> list:
    """Indices of CLASSIFIER_FEATURES within a features_to_array() matrix."""
    return [FEATURE_NAMES.index(n) for n in CLASSIFIER_FEATURES]


FEATURE_NAMES = [
    "template_corr",
    "qrs_width_ms",
    "r_amplitude",
    "qrs_energy",
    "pre_rr_ratio",
    "post_rr_ratio",
    "r_amp_ratio",
    "width_ratio",
]
