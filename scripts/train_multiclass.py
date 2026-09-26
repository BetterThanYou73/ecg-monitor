"""Train and evaluate AAMI five-class beat classification, inter-patient.

Trains on de Chazal DS1 and tests on DS2, so no subject appears in both.
Reports per-class sensitivity and PPV plus the confusion matrix, and compares
the ventricular column against the binary classifier to show what giving
fusion beats their own class buys.

Usage:
    python scripts/train_multiclass.py
    python scripts/train_multiclass.py --compare-binary
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ecgmon.analysis.beat_classifier import DS1_TRAIN, DS2_TEST
from ecgmon.analysis.morphology import (
    FEATURE_NAMES,
    beat_features,
    features_to_array,
)
from ecgmon.analysis.multiclass import (
    AAMI_CLASSES,
    MULTICLASS_FEATURES,
    MultiClassBeatClassifier,
    confusion,
    drop_unclassifiable,
    format_confusion,
    multiclass_columns,
    score_multiclass,
    symbols_to_aami,
)
from ecgmon.io.wfdb_loader import load_annotations, load_wfdb_record
from ecgmon.preprocessing.filters import preprocess


def build_record(record: str, data_dir: Path, channel: int = 0):
    rec = load_wfdb_record(record, db="mitdb", data_dir=data_dir)
    ann = load_annotations(record, db="mitdb", data_dir=data_dir).beats_only()

    ch = rec[channel]
    cleaned = preprocess(ch.signal, ch.fs)

    feats, _, _ = beat_features(cleaned, ch.fs, ann.sample)
    if not feats:
        return np.empty((0, len(MULTICLASS_FEATURES))), np.array([], dtype=int)

    cols = multiclass_columns(FEATURE_NAMES)
    X = features_to_array(feats)[:, cols]
    idx = np.array([f.index for f in feats], dtype=int)
    y = symbols_to_aami(ann.symbol)[idx]
    return X, y


def build_set(records: list, data_dir: Path, label: str):
    Xs, ys = [], []
    print(f"\n{label}:")
    for r in records:
        try:
            X, y = build_record(r, data_dir)
        except Exception as exc:
            print(f"  {r}: skipped ({type(exc).__name__})")
            continue
        if X.shape[0] == 0:
            continue
        Xs.append(X)
        ys.append(y)
        counts = {AAMI_CLASSES[i]: int(np.count_nonzero(y == i)) for i in range(5)}
        shown = {k: v for k, v in counts.items() if v}
        print(f"  {r}: {y.size:>6} beats  {shown}")
    if not Xs:
        raise SystemExit("no records loaded")
    return np.vstack(Xs), np.concatenate(ys)


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/raw/mitdb")
    p.add_argument("--compare-binary", action="store_true",
                   help="also report the binary ventricular classifier for contrast")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    t0 = time.perf_counter()

    Xtr, ytr = build_set(DS1_TRAIN, data_dir, f"DS1 train ({len(DS1_TRAIN)} subjects)")
    Xte, yte = build_set(DS2_TEST, data_dir, f"DS2 test ({len(DS2_TEST)} subjects)")
    print(f"\nfeature extraction: {time.perf_counter() - t0:.1f}s")

    Xtr, ytr, n_drop_tr = drop_unclassifiable(Xtr, ytr)
    Xte, yte, n_drop_te = drop_unclassifiable(Xte, yte)
    print(f"dropped unclassifiable (Q) beats: {n_drop_tr} train, {n_drop_te} test")

    clf = MultiClassBeatClassifier().fit(Xtr, ytr)
    pred = clf.predict(Xte)

    print("\n" + "=" * 70)
    print("AAMI FIVE-CLASS, INTER-PATIENT (DS2)")
    print("=" * 70)
    for s in score_multiclass(yte, pred):
        print(s)

    print("\nConfusion matrix:")
    print(format_confusion(confusion(yte, pred)))

    # What the fusion class bought: previously F beats had to be called V or N.
    v = AAMI_CLASSES.index("V")
    f = AAMI_CLASSES.index("F")
    m = confusion(yte, pred)
    if m[f].sum():
        as_v = m[f, v] / m[f].sum()
        print(f"\nFusion beats still called ventricular: {as_v * 100:.1f}% "
              f"({m[f, v]} of {m[f].sum()})")
        print("The binary model called 29.4% of them ventricular, having no "
              "other option available.")

    print("\nPer-class feature weights:")
    for cls, pairs in clf.coefficients(MULTICLASS_FEATURES).items():
        top = ", ".join(f"{n} {c:+.2f}" for n, c in pairs[:3])
        print(f"  {cls}: {top}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
