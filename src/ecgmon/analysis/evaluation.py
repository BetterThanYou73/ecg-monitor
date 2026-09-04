"""Detector evaluation against reference annotations.

Follows the ANSI/AAMI EC57 convention: a detection counts as correct if it
falls within a fixed tolerance of a reference beat, each reference beat may
be matched at most once, and sensitivity and positive predictive value are
reported rather than plain accuracy.

Accuracy is the wrong measure here and it is worth being explicit about why.
Beats vastly outnumber non-beats, so a detector can miss a clinically
important fraction of them and still post a high accuracy. Sensitivity says
what fraction of real beats were found; PPV says what fraction of detections
were real. Both matter, and they trade against each other.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# EC57 matching tolerance.
DEFAULT_TOLERANCE_S = 0.15


@dataclass
class DetectionScore:
    """Match statistics for one record."""

    record_id: str
    n_reference: int
    n_detected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    tolerance_s: float
    mean_abs_error_ms: float  # placement error on matched beats
    std_abs_error_ms: float

    @property
    def sensitivity(self) -> float:
        """Fraction of reference beats that were detected (recall)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else float("nan")

    @property
    def ppv(self) -> float:
        """Fraction of detections that were real beats (precision)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        se, pp = self.sensitivity, self.ppv
        if not np.isfinite(se) or not np.isfinite(pp) or (se + pp) == 0:
            return float("nan")
        return 2 * se * pp / (se + pp)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(sensitivity=self.sensitivity, ppv=self.ppv, f1=self.f1)
        return d

    def __str__(self) -> str:
        return (
            f"{self.record_id:>8}  ref={self.n_reference:>6}  det={self.n_detected:>6}  "
            f"TP={self.true_positives:>6}  FP={self.false_positives:>5}  "
            f"FN={self.false_negatives:>5}  "
            f"Se={self.sensitivity * 100:6.2f}%  PPV={self.ppv * 100:6.2f}%  "
            f"F1={self.f1 * 100:6.2f}%  err={self.mean_abs_error_ms:5.1f}ms"
        )


def match_detections(
    detected: np.ndarray,
    reference: np.ndarray,
    fs: float,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedily pair detections with reference beats.

    Both sequences are sorted and swept once together, so this is O(n) rather
    than the O(n*m) a naive nearest-search would cost -- which matters at
    24-hour scale, where a record holds on the order of 100,000 beats.

    Returns:
        ``(matched_det, matched_ref, unmatched_det, unmatched_ref)`` as index
        arrays into the inputs.
    """
    detected = np.sort(np.asarray(detected, dtype=np.int64))
    reference = np.sort(np.asarray(reference, dtype=np.int64))
    tol = tolerance_s * fs

    matched_det: list[int] = []
    matched_ref: list[int] = []
    used_ref = np.zeros(reference.size, dtype=bool)

    j = 0
    for i, d in enumerate(detected):
        # Advance past reference beats that are already too far behind.
        while j < reference.size and reference[j] < d - tol:
            j += 1
        # Among references within tolerance, take the closest unused one.
        best_k, best_dist = -1, None
        k = j
        while k < reference.size and reference[k] <= d + tol:
            if not used_ref[k]:
                dist = abs(int(reference[k]) - int(d))
                if best_dist is None or dist < best_dist:
                    best_k, best_dist = k, dist
            k += 1
        if best_k >= 0:
            used_ref[best_k] = True
            matched_det.append(i)
            matched_ref.append(best_k)

    unmatched_det = np.setdiff1d(np.arange(detected.size), np.array(matched_det, dtype=int))
    unmatched_ref = np.flatnonzero(~used_ref)
    return (
        np.array(matched_det, dtype=int),
        np.array(matched_ref, dtype=int),
        unmatched_det,
        unmatched_ref,
    )


def score_detections(
    detected: np.ndarray,
    reference: np.ndarray,
    fs: float,
    record_id: str = "",
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> DetectionScore:
    """Score a detector's output against reference beats."""
    detected = np.sort(np.asarray(detected, dtype=np.int64))
    reference = np.sort(np.asarray(reference, dtype=np.int64))

    m_det, m_ref, u_det, u_ref = match_detections(
        detected, reference, fs, tolerance_s=tolerance_s
    )

    if m_det.size:
        errs = np.abs(detected[m_det] - reference[m_ref]) / fs * 1000.0
        mean_err, std_err = float(np.mean(errs)), float(np.std(errs))
    else:
        mean_err = std_err = float("nan")

    return DetectionScore(
        record_id=record_id,
        n_reference=int(reference.size),
        n_detected=int(detected.size),
        true_positives=int(m_det.size),
        false_positives=int(u_det.size),
        false_negatives=int(u_ref.size),
        tolerance_s=tolerance_s,
        mean_abs_error_ms=mean_err,
        std_abs_error_ms=std_err,
    )


def aggregate(scores: list[DetectionScore]) -> DetectionScore:
    """Pool scores across records (gross statistics, beat-weighted).

    Beat-weighted rather than an average of per-record rates, so a short
    record cannot count as much as a long one.
    """
    if not scores:
        raise ValueError("no scores to aggregate")
    tp = sum(s.true_positives for s in scores)
    fp = sum(s.false_positives for s in scores)
    fn = sum(s.false_negatives for s in scores)
    errs = [s.mean_abs_error_ms for s in scores if np.isfinite(s.mean_abs_error_ms)]
    return DetectionScore(
        record_id=f"ALL({len(scores)})",
        n_reference=sum(s.n_reference for s in scores),
        n_detected=sum(s.n_detected for s in scores),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        tolerance_s=scores[0].tolerance_s,
        mean_abs_error_ms=float(np.mean(errs)) if errs else float("nan"),
        std_abs_error_ms=float(np.mean([s.std_abs_error_ms for s in scores if np.isfinite(s.std_abs_error_ms)]))
        if errs
        else float("nan"),
    )


def header() -> str:
    """Column header matching ``DetectionScore.__str__``."""
    return (
        f"{'record':>8}  {'ref':>10}  {'det':>10}  {'TP':>9}  {'FP':>8}  {'FN':>8}  "
        f"{'Se':>9}  {'PPV':>10}  {'F1':>9}  {'err':>8}"
    )
