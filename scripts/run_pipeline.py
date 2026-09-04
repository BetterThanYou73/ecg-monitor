"""End-to-end pipeline on one record: ingest -> quality -> filter -> detect
-> RR/HRV -> ectopic screen -> figures.

Runs the full Weeks 1-2 chain and writes figures to outputs/. Both channels
of the MIT-BIH record are carried through as separate sensors, so the
multi-channel path is exercised rather than just designed.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --record 106 --window 12 22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ecgmon.analysis.evaluation import score_detections
from ecgmon.analysis.rr import (
    ectopic_candidates,
    heart_rate_trend,
    hrv_metrics,
    rr_from_peaks,
)
from ecgmon.detection.rpeaks import detect_rpeaks
from ecgmon.io.wfdb_loader import (
    VENTRICULAR_BEATS,
    load_annotations,
    load_wfdb_record,
)
from ecgmon.preprocessing.filters import preprocess_channel
from ecgmon.preprocessing.quality import channel_sqi, rank_channels
from ecgmon.viz.plots import (
    plot_detection_overview,
    plot_hr_trend,
    plot_multichannel,
)


def rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", default="106",
                   help="MIT-BIH record id (default 106: PVC-rich)")
    p.add_argument("--data-dir", default="data/raw/mitdb")
    p.add_argument("--window", nargs=2, type=float, default=[0.0, 10.0],
                   metavar=("START", "END"), help="plot window in seconds")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--powerline-hz", type=float, default=60.0)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    powerline = None if args.powerline_hz == 0 else args.powerline_hz
    win = (args.window[0], args.window[1])

    # ---------------------------------------------------------------- ingest
    rule(f"1. Ingest - record {args.record}")
    rec = load_wfdb_record(args.record, db="mitdb", data_dir=data_dir)
    print(rec.summary())
    print(f"\nuniform sampling rate: {rec.has_uniform_fs}  "
          f"(rates: {sorted(rec.sampling_rates)})")
    print(f"single channel: {rec.is_single_channel}")

    # ------------------------------------------------------- signal quality
    rule("2. Signal quality per sensor")
    for sensor_id, sqi in rank_channels(rec):
        ch = rec[sensor_id]
        flag = "usable" if sqi >= 0.5 else "POOR"
        print(f"  {sensor_id:<14} {ch.position:<12} median SQI = {sqi:.3f}  [{flag}]")

    # ------------------------------------------------------------ filtering
    rule("3. Preprocessing")
    print(f"  baseline-wander highpass 0.5 Hz, "
          f"{'notch ' + str(powerline) + ' Hz, ' if powerline else ''}lowpass 40 Hz")
    clean = rec.map_channels(
        lambda ch: preprocess_channel(ch, powerline_hz=powerline)
    )
    for ch in clean:
        raw = rec[ch.sensor_id]
        print(f"  {ch.sensor_id:<14} SQI {channel_sqi(raw):.3f} -> {channel_sqi(ch):.3f}")

    # ------------------------------------------------------------ detection
    rule("4. R-peak detection (Pan-Tompkins)")
    peaks_by_sensor: dict[str, np.ndarray] = {}
    stages_by_sensor: dict[str, dict] = {}
    for ch in clean:
        pk, stages = detect_rpeaks(ch.signal, ch.fs, return_stages=True)
        peaks_by_sensor[ch.sensor_id] = pk
        stages_by_sensor[ch.sensor_id] = stages
        mean_hr = 60.0 / np.mean(np.diff(pk) / ch.fs) if pk.size > 1 else float("nan")
        print(f"  {ch.sensor_id:<14} {pk.size:>5} beats   mean HR {mean_hr:5.1f} bpm")

    # Cross-sensor agreement. With MIT-BIH leads this is an easy case, but it
    # is the same computation that will judge real sensors against each other.
    if clean.n_channels > 1:
        ids = [ch.sensor_id for ch in clean]
        a, b = ids[0], ids[1]
        agree = score_detections(
            peaks_by_sensor[b], peaks_by_sensor[a], clean[a].fs, record_id=f"{b} vs {a}"
        )
        print(f"\n  cross-sensor agreement ({b} against {a}):")
        print(f"    {agree.true_positives} common beats, "
              f"{agree.false_positives} only in {b}, "
              f"{agree.false_negatives} only in {a}")
        print(f"    concordance F1 = {agree.f1 * 100:.2f}%, "
              f"mean offset {agree.mean_abs_error_ms:.1f} ms")

    # -------------------------------------------------- validate vs reference
    rule("5. Validation against reference annotations")
    try:
        ann = load_annotations(args.record, db="mitdb", data_dir=data_dir).beats_only()
        primary = clean[0]
        score = score_detections(
            peaks_by_sensor[primary.sensor_id], ann.sample, primary.fs,
            record_id=args.record,
        )
        print(f"  reference beats: {len(ann)}   symbols: {ann.counts()}")
        print(f"  Se = {score.sensitivity * 100:.2f}%   "
              f"PPV = {score.ppv * 100:.2f}%   F1 = {score.f1 * 100:.2f}%")
        print(f"  placement error = {score.mean_abs_error_ms:.1f} "
              f"+/- {score.std_abs_error_ms:.1f} ms")
        n_pvc = ann.ventricular.size
        print(f"  ventricular beats in reference: {n_pvc}")
    except Exception as exc:
        ann = None
        n_pvc = 0
        print(f"  annotations unavailable ({type(exc).__name__}: {exc})")

    # --------------------------------------------------------- RR / HRV
    rule("6. RR intervals and HRV")
    primary = clean[0]
    series = rr_from_peaks(peaks_by_sensor[primary.sensor_id], primary.fs)
    rejected = int(np.count_nonzero(~series.valid))
    print(f"  intervals: {series.rr_s.size}  (rejected as implausible: {rejected})")
    m = hrv_metrics(series)
    print(f"  mean HR   {m.mean_hr_bpm:6.1f} bpm   "
          f"(min {m.min_hr_bpm:.1f}, max {m.max_hr_bpm:.1f})")
    print(f"  mean RR   {m.mean_rr_ms:6.1f} ms")
    print(f"  SDNN      {m.sdnn_ms:6.1f} ms")
    print(f"  RMSSD     {m.rmssd_ms:6.1f} ms")
    print(f"  pNN50     {m.pnn50 * 100:6.1f} %")
    print(f"  CV(RR)    {m.cv_rr:6.3f}")

    # ------------------------------------------------------- ectopic screen
    rule("7. Ectopic-beat screen (rhythm only)")
    cand = ectopic_candidates(series)
    print(f"  short-then-long candidates: {cand.size}")
    if ann is not None and n_pvc:
        print(f"  reference ventricular beats: {n_pvc}")
        print("  NOTE: a rhythm-only screen. Morphology classification is the")
        print("        next milestone; this narrows the search, it does not")
        print("        diagnose. Counts are not expected to match.")

    # -------------------------------------------------------------- figures
    rule("8. Figures")
    p1 = plot_detection_overview(
        primary,
        peaks_by_sensor[primary.sensor_id],
        stages=stages_by_sensor[primary.sensor_id],
        window_s=win,
        out_path=out_dir / f"{args.record}_detection.png",
    )
    print(f"  {p1}")

    p2 = plot_multichannel(
        clean,
        peaks_by_sensor=peaks_by_sensor,
        window_s=win,
        out_path=out_dir / f"{args.record}_multichannel.png",
    )
    print(f"  {p2}")

    centres, hr = heart_rate_trend(series, window_s=60.0)
    if centres.size:
        p3 = plot_hr_trend(
            centres, hr,
            out_path=out_dir / f"{args.record}_hr_trend.png",
            title=f"Record {args.record}: heart rate, 60 s windows",
        )
        print(f"  {p3}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
