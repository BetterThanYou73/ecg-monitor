"""Benchmark R-peak detection against MIT-BIH reference annotations.

Usage:
    python scripts/evaluate_rpeaks.py                  # default record set
    python scripts/evaluate_rpeaks.py --records 100 106 119 --method neurokit
    python scripts/evaluate_rpeaks.py --all            # all 48 MIT-BIH records

Records are cached under data/raw/mitdb on first use, so repeat runs work
without network access.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ecgmon.analysis.evaluation import aggregate, header, score_detections
from ecgmon.detection.rpeaks import detect_rpeaks, detect_rpeaks_neurokit
from ecgmon.io.wfdb_loader import load_annotations, load_wfdb_record
from ecgmon.preprocessing.filters import preprocess

# A deliberately mixed default set rather than the easy records only:
# 100/103 are clean, 106/119 are PVC-heavy, 108 has notoriously noisy
# baseline, 203 has both noise and ectopy, 208 has many ventricular beats.
DEFAULT_RECORDS = ["100", "103", "106", "108", "119", "203", "208", "222"]

ALL_MITDB = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def evaluate_record(
    record: str,
    data_dir: Path,
    method: str,
    powerline_hz: float | None,
    channel: int,
    recover: bool = False,
) -> tuple:
    """Return (score, elapsed_seconds) for one record."""
    rec = load_wfdb_record(record, db="mitdb", data_dir=data_dir)
    ann = load_annotations(record, db="mitdb", data_dir=data_dir).beats_only()

    ch = rec[channel]
    cleaned = preprocess(ch.signal, ch.fs, powerline_hz=powerline_hz)

    start = time.perf_counter()
    if method == "pan_tompkins":
        peaks = detect_rpeaks(cleaned, ch.fs, recover_missed=recover)
    elif method == "neurokit":
        peaks = detect_rpeaks_neurokit(cleaned, ch.fs)
    else:
        raise ValueError(f"unknown method {method!r}")
    elapsed = time.perf_counter() - start

    score = score_detections(peaks, ann.sample, ch.fs, record_id=record)
    return score, elapsed, ch.duration_s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", default=None,
                   help="MIT-BIH record ids (default: a mixed 8-record set)")
    p.add_argument("--all", action="store_true", help="use all 48 MIT-BIH records")
    p.add_argument("--method", default="pan_tompkins",
                   choices=["pan_tompkins", "neurokit"])
    p.add_argument("--channel", type=int, default=0,
                   help="signal index within the record (default: 0, usually MLII)")
    p.add_argument("--data-dir", default="data/raw/mitdb")
    p.add_argument("--powerline-hz", type=float, default=60.0,
                   help="mains notch frequency; pass 0 to disable")
    p.add_argument("--recover", action="store_true",
                   help="enable the gap-filling second pass (raises Se, lowers PPV)")
    p.add_argument("--json-out", default=None, help="write per-record scores as JSON")
    args = p.parse_args(argv)

    records = ALL_MITDB if args.all else (args.records or DEFAULT_RECORDS)
    data_dir = Path(args.data_dir)
    powerline = None if args.powerline_hz == 0 else args.powerline_hz

    print(f"\nR-peak detection benchmark - method={args.method}, "
          f"channel={args.channel}, recover={args.recover}, {len(records)} record(s)")
    print(f"data: {data_dir.resolve()}\n")
    print(header())
    print("-" * len(header()))

    scores, total_elapsed, total_signal = [], 0.0, 0.0
    for rec_id in records:
        try:
            score, elapsed, dur = evaluate_record(
                rec_id, data_dir, args.method, powerline, args.channel, args.recover
            )
        except Exception as exc:  # keep going: one bad record should not abort a sweep
            print(f"{rec_id:>8}  FAILED: {type(exc).__name__}: {exc}")
            continue
        scores.append(score)
        total_elapsed += elapsed
        total_signal += dur
        print(score)

    if not scores:
        print("\nNo records evaluated.")
        return 1

    print("-" * len(header()))
    total = aggregate(scores)
    print(total)

    speed = total_signal / total_elapsed if total_elapsed else float("nan")
    print(
        f"\nProcessed {total_signal / 3600:.2f} h of ECG in {total_elapsed:.2f} s "
        f"({speed:,.0f}x realtime)."
    )
    # A 24-hour recording is the target; report whether that is tractable.
    if np.isfinite(speed) and speed > 0:
        print(f"Extrapolated: 24 h single channel -> {24 * 3600 / speed:.1f} s.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"method": args.method, "per_record": [s.to_dict() for s in scores],
                 "aggregate": total.to_dict()},
                indent=2,
            )
        )
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
