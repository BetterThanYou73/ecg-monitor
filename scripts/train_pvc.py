"""Train and evaluate ventricular-beat classification on MIT-BIH.

Reports the inter-patient result (train and test on different subjects) as the
headline, and the intra-patient result alongside it to show how much the
easier protocol inflates the number.

Features are computed at the reference annotation positions, which isolates
classification performance from R-peak detection error. Pass --detected to
run the realistic end-to-end version instead, where beats come from the
detector and unmatched detections count against the result.

Usage:
    python scripts/train_pvc.py
    python scripts/train_pvc.py --detected
    python scripts/train_pvc.py --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ecgmon.analysis.beat_classifier import (
    DS1_TRAIN,
    DS2_TEST,
    VentricularClassifier,
    score_classification,
)
from ecgmon.analysis.evaluation import match_detections
from ecgmon.analysis.morphology import (
    CLASSIFIER_FEATURES,
    FEATURE_NAMES,
    beat_features,
    classifier_columns,
    features_to_array,
)
from ecgmon.detection.rpeaks import detect_rpeaks
from ecgmon.io.wfdb_loader import (
    VENTRICULAR_BEATS,
    load_annotations,
    load_wfdb_record,
)
from ecgmon.preprocessing.filters import preprocess

MATCH_TOLERANCE_S = 0.15


def build_record(record: str, data_dir: Path, use_detected: bool, channel: int = 0):
    """Features and labels for one record.

    Returns ``(X, y, n_missed)``. ``n_missed`` counts annotated ventricular
    beats the detector failed to find, which the end-to-end result must
    account for -- a beat that was never detected cannot be classified, and
    ignoring it would flatter the result.
    """
    rec = load_wfdb_record(record, db="mitdb", data_dir=data_dir)
    ann = load_annotations(record, db="mitdb", data_dir=data_dir).beats_only()

    ch = rec[channel]
    cleaned = preprocess(ch.signal, ch.fs)

    is_v = np.isin(ann.symbol, list(VENTRICULAR_BEATS)).astype(int)

    if not use_detected:
        peaks = ann.sample
        labels = is_v
        n_missed = 0
    else:
        peaks = detect_rpeaks(cleaned, ch.fs)
        m_det, m_ref, u_det, _ = match_detections(
            peaks, ann.sample, ch.fs, tolerance_s=MATCH_TOLERANCE_S
        )
        # Label each detection by the reference beat it matched; unmatched
        # detections are false beats and are labelled non-ventricular.
        labels = np.zeros(peaks.size, dtype=int)
        order_ref = np.argsort(ann.sample)
        labels[m_det] = is_v[order_ref][m_ref]
        n_missed = int(np.count_nonzero(is_v) - np.count_nonzero(labels == 1))
        n_missed = max(0, n_missed)

    feats, _, _ = beat_features(cleaned, ch.fs, np.sort(peaks))
    if not feats:
        return np.empty((0, len(CLASSIFIER_FEATURES))), np.array([], dtype=int), n_missed

    X = features_to_array(feats)[:, classifier_columns()]
    idx = np.array([f.index for f in feats], dtype=int)
    y = labels[idx]
    return X, y, n_missed


def select_threshold(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     n_folds: int = 4) -> float:
    """Pick a decision threshold using subject-wise folds inside the training set.

    Subjects are held out whole, mirroring the inter-patient protocol, so the
    chosen threshold reflects performance on unseen people rather than on
    unseen beats from people already learned.
    """
    subjects = np.unique(groups)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(subjects)
    folds = np.array_split(shuffled, n_folds)

    grid = np.arange(0.30, 0.96, 0.02)
    f1_sum = np.zeros(grid.size)
    used = 0

    for held in folds:
        val = np.isin(groups, held)
        if not val.any() or val.all():
            continue
        if np.count_nonzero(y[~val] == 1) < 5 or np.count_nonzero(y[val] == 1) < 5:
            continue
        fold_clf = VentricularClassifier().fit(X[~val], y[~val])
        proba = fold_clf.predict_proba(X[val])
        for i, t in enumerate(grid):
            s = score_classification(y[val], (proba >= t).astype(int))
            f1_sum[i] += 0.0 if not np.isfinite(s.f1) else s.f1
        used += 1

    if used == 0:
        return 0.5
    return float(grid[int(np.argmax(f1_sum))])


def build_set(records: list[str], data_dir: Path, use_detected: bool):
    Xs, ys, groups, missed = [], [], [], 0
    for r in records:
        try:
            X, y, n_missed = build_record(r, data_dir, use_detected)
        except Exception as exc:
            print(f"  {r}: skipped ({type(exc).__name__}: {exc})")
            continue
        if X.shape[0] == 0:
            continue
        Xs.append(X)
        ys.append(y)
        groups.append(np.full(y.size, int(r)))
        missed += n_missed
        print(f"  {r}: {y.size:>6} beats, {int(np.count_nonzero(y)):>5} ventricular")
    if not Xs:
        raise SystemExit("no records loaded")
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(groups), missed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/raw/mitdb")
    p.add_argument("--detected", action="store_true",
                   help="use detector output instead of reference beat positions")
    p.add_argument("--threshold", type=float, default=None,
                   help="fixed decision threshold; default selects it on DS1")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    mode = "detected peaks (end-to-end)" if args.detected else "reference peaks"

    print(f"\nVentricular-beat classification - {mode}")
    print("=" * 70)

    t0 = time.perf_counter()
    print(f"\nDS1 (train, {len(DS1_TRAIN)} subjects):")
    Xtr, ytr, gtr, missed_tr = build_set(DS1_TRAIN, data_dir, args.detected)
    print(f"\nDS2 (test, {len(DS2_TEST)} subjects):")
    Xte, yte, gte, missed_te = build_set(DS2_TEST, data_dir, args.detected)
    print(f"\nfeature extraction: {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------- threshold selection
    # Chosen on training subjects only. Picking it on DS2 would tune the model
    # to the test set and report a number that cannot be reproduced on new
    # patients -- the same leak the inter-patient split exists to prevent.
    if args.threshold is not None:
        threshold = args.threshold
        print(f"\nUsing fixed threshold {threshold:.2f}")
    else:
        threshold = select_threshold(Xtr, ytr, gtr)
        print(f"\nThreshold selected on held-out DS1 subjects: {threshold:.2f}")

    # --------------------------------------------- controlled protocol study
    # Both models are scored on exactly the same beats. Each DS2 subject's
    # beats are split in half: the second half is the shared test set, and the
    # first half is given to the intra-patient model as training data. The
    # only difference between the two models is therefore whether they have
    # already seen beats from the people they are being tested on.
    #
    # Comparing a DS2-only test set against a random split of all beats would
    # not isolate that: the two test sets would differ in size, subject mix
    # and class balance, and the comparison would measure those instead.
    rng = np.random.default_rng(0)
    first_half = np.zeros(yte.size, dtype=bool)
    for subj in np.unique(gte):
        idx = np.flatnonzero(gte == subj)
        rng.shuffle(idx)
        first_half[idx[: len(idx) // 2]] = True
    held_out = ~first_half

    clf = VentricularClassifier().fit(Xtr, ytr)
    inter = score_classification(
        yte[held_out],
        clf.predict(Xte[held_out], threshold=threshold),
        name="INTER-patient",
    )

    X_intra = np.vstack([Xtr, Xte[first_half]])
    y_intra = np.concatenate([ytr, yte[first_half]])
    g_intra = np.concatenate([gtr, gte[first_half]])
    intra_threshold = select_threshold(X_intra, y_intra, g_intra)
    clf_intra = VentricularClassifier().fit(X_intra, y_intra)
    intra = score_classification(
        yte[held_out],
        clf_intra.predict(Xte[held_out], threshold=intra_threshold),
        name="intra-patient",
    )

    print("\n" + "=" * 70)
    print("RESULTS (ventricular class)")
    print("=" * 70)
    print(intra)
    print(inter)

    gap = intra.f1 - inter.f1
    print(f"\nInflation from ignoring subject identity: {gap * 100:.2f} F1 points.")
    if args.detected:
        print(f"Ventricular beats never detected (not classifiable): {missed_te}")
        total_v = inter.n_ventricular + missed_te
        if total_v:
            eff = inter.true_positives / total_v
            print(f"End-to-end ventricular sensitivity incl. detection misses: "
                  f"{eff * 100:.2f}%")

    print("\nFeature weights (standardised, |largest| first):")
    for name, coef in clf.coefficients(CLASSIFIER_FEATURES):
        direction = "higher -> ventricular" if coef > 0 else "lower  -> ventricular"
        print(f"  {name:<16} {coef:+7.3f}   {direction}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
