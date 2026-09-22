"""Build the monitoring dashboard for one recording.

Runs the whole chain -- ingest, quality, filter, detect, classify, summarise --
and writes a self-contained HTML report.

The classifier is trained on DS1 subjects and applied to the record being
reported, so the record is never used to fit the model that judges it. If the
record happens to be a DS1 member the training set drops it first.

Usage:
    python scripts/build_dashboard.py --record 106
    python scripts/build_dashboard.py --record 119 --out outputs/119.html
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ecgmon.analysis.beat_classifier import DS1_TRAIN, VentricularClassifier
from ecgmon.analysis.morphology import (
    beat_features,
    classifier_columns,
    features_to_array,
)
from ecgmon.analysis.summary import (
    representative_beats,
    summarize_channel,
)
from ecgmon.detection.rpeaks import detect_rpeaks
from ecgmon.io.wfdb_loader import load_wfdb_record
from ecgmon.preprocessing.filters import preprocess_channel
from ecgmon.preprocessing.quality import assess_channel
from ecgmon.viz.dashboard import build_dashboard

sys.path.insert(0, str(Path(__file__).parent))


def train_classifier(data_dir: Path, exclude: str) -> VentricularClassifier:
    """Fit on DS1, excluding the record being reported."""
    from train_pvc import build_set, select_threshold

    records = [r for r in DS1_TRAIN if r != exclude]
    X, y, g, _ = build_set(records, data_dir, use_detected=False)
    threshold = select_threshold(X, y, g)
    clf = VentricularClassifier().fit(X, y)
    return clf, threshold


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", default="106")
    p.add_argument("--data-dir", default="data/raw/mitdb")
    p.add_argument("--out", default=None)
    p.add_argument("--window", nargs=2, type=float, default=[0.0, 10.0],
                   metavar=("START", "END"))
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else Path("outputs") / f"dashboard_{args.record}.html"

    t0 = time.perf_counter()

    print(f"Loading {args.record}...")
    rec = load_wfdb_record(args.record, db="mitdb", data_dir=data_dir)
    print(rec.summary())

    print("\nTraining classifier on DS1 (excluding this record)...")
    clf, threshold = train_classifier(data_dir, args.record)
    print(f"  threshold {threshold:.2f}")

    summaries: dict = {}
    peaks_by_sensor: dict = {}
    examples: list = []

    for i, raw in enumerate(rec):
        print(f"\nProcessing {raw.sensor_id}...")
        clean = preprocess_channel(raw, powerline_hz=60.0)
        peaks = detect_rpeaks(clean.signal, clean.fs)
        peaks_by_sensor[raw.sensor_id] = peaks
        print(f"  {peaks.size} beats")

        feats, _, _ = beat_features(clean.signal, clean.fs, peaks)
        pvc_samples = np.array([], dtype=int)
        if feats:
            X = features_to_array(feats)[:, classifier_columns()]
            pred = clf.predict(X, threshold=threshold)
            pvc_samples = np.array(
                [f.sample for f, flag in zip(feats, pred) if flag], dtype=int
            )
        print(f"  {pvc_samples.size} classified ventricular")

        sqi = assess_channel(raw, window_s=10.0)
        summaries[raw.sensor_id] = summarize_channel(
            raw, peaks,
            event_samples={"PVC": pvc_samples},
            sqi_windows=sqi,
        )

        if i == 0:
            examples = representative_beats(clean.signal, clean.fs, pvc_samples, n=6)

    print()
    for s in summaries.values():
        print(s.text_report())

    caveats = [
        "PVC labels come from an automatic classifier scored at 92% sensitivity "
        "and 77% precision on unseen subjects. Roughly one in four flagged beats "
        "is expected to be a false positive.",
        "Rates are per analysed hour, not per elapsed hour. Stretches excluded "
        "for signal dropout are not counted in the denominator.",
        "Research prototype on public data. Not validated on the target sensor "
        "hardware and not for diagnostic use.",
    ]

    path = build_dashboard(
        rec,
        summaries,
        peaks_by_sensor,
        event_label="PVC",
        examples=examples,
        out_path=out,
        title=f"ECG monitoring report — record {args.record}",
        subtitle=(
            f"MIT-BIH {args.record} · {rec.n_channels} sensor(s) · "
            f"{rec[0].duration_s / 3600:.2f} h · classifier trained on other subjects"
        ),
        caveats=caveats,
        trace_window_s=(args.window[0], args.window[1]),
    )

    size_kb = path.stat().st_size / 1024
    print(f"\nWrote {path} ({size_kb:.0f} KB) in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
