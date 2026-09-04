# ECG-Monitor

Signal-analysis pipeline for a portable long-term ECG monitoring system.

The target hardware is a set of **independent single-channel ECG sensors**
placed at different positions around the chest, recording **independently but
synchronously**. This is not a clinical 12-lead system — the electrode
configuration and lead geometry are different — so it is treated as a
distributed, synchronised, multi-position sensing system.

The consequence for the software, and the single constraint that shapes this
codebase: **a single-sensor recording is the one-channel case of a
multi-sensor recording.** Timestamps, sensor identity and per-sensor clock
offsets are in the data model from the start, so extending to several
synchronised sensors needs no rewrite.

---

## Status

Weeks 1–2 of the initial development cycle: background review and baseline
pipeline.

| Milestone | State |
|---|---|
| PhysioNet / WFDB dataset ingestion | done |
| Preprocessing (baseline wander, mains, band-limiting) | done |
| ECG visualisation | done |
| QRS / R-peak detection | done |
| Baseline feature extraction (RR, HR, HRV) | done |
| Signal-quality assessment | done |
| Detector validation against annotations | done |
| Beat-morphology classification | not started (weeks 3–4) |
| Arrhythmia classification | not started (weeks 3–4) |

---

## Results

R-peak detection against MIT-BIH reference annotations, 8 records,
18,552 annotated beats, 4.0 hours of ECG. Matching follows ANSI/AAMI EC57
with a 150 ms tolerance and one-to-one pairing.

| Detector | Lead | Sensitivity | PPV | F1 |
|---|---|---|---|---|
| Pan-Tompkins (this repo) | 0 (mostly MLII) | 98.37% | 99.27% | **98.82%** |
| Pan-Tompkins + gap recovery | 0 | 99.50% | 97.96% | 98.72% |
| NeuroKit2 (reference) | 0 | 97.48% | 98.73% | 98.10% |
| Pan-Tompkins | 1 (V1 etc.) | 83.96% | 97.27% | 90.13% |
| Pan-Tompkins + gap recovery | 1 | 89.36% | 94.89% | 92.04% |

Throughput is roughly 26,000× realtime, so a 24-hour single-channel recording
processes in about 3 seconds. Continuous 24-hour monitoring is not
computationally constrained.

Sensitivity and PPV are reported rather than accuracy: beats vastly outnumber
non-beats, so accuracy stays high even when a clinically significant fraction
of beats is missed.

### The finding that matters for multi-sensor work

**Detection accuracy depends strongly on sensor position.** The same detector,
on the same beats, in the same recordings, scores 98.4% sensitivity on lead 0
and 84.0% on lead 1.

The cause is diagnosed, not guessed. In records with frequent ectopic beats
(106, 208), the secondary lead shows large ectopic beats alongside small
normal ones. Pan-Tompkins tracks a single running signal-peak estimate, so the
large beats pull the threshold above the small ones and normal beats are lost
in runs. On record 106 lead V1 this costs 62 percentage points of sensitivity.

This bears directly on the project's research question of whether some sensor
positions detect particular abnormalities better than others. The answer so
far is that they demonstrably differ — but part of that difference is the
detector, not the physiology, and the two have to be separated before any
claim about optimal placement can be made. A position-independent detector is
a prerequisite for the fusion work, not an optimisation of it.

`--recover` enables a second pass that re-scans long gaps at a locally derived
threshold. It trades PPV for sensitivity and is off by default, because
inventing beats inside a genuine pause corrupts the rhythm analysis that the
pause feeds into.

---

## Layout

```
src/ecgmon/
  io/record.py         data model: ChannelRecord, SynchronizedRecording
  io/wfdb_loader.py    PhysioNet/WFDB ingestion and beat annotations
  preprocessing/
    filters.py         baseline wander, mains notch, bandpass, resampling
    quality.py         windowed signal-quality index, sensor ranking
  detection/rpeaks.py  Pan-Tompkins with inspectable stages, gap recovery
  analysis/
    rr.py              RR intervals, HR trend, time-domain HRV, ectopic screen
    evaluation.py      EC57 matching, sensitivity/PPV/F1
  viz/plots.py         detection overview, synchronised multi-channel, HR trend
scripts/
  fetch_data.py        cache PhysioNet records locally (retries 502s)
  run_pipeline.py      full chain on one record, writes figures
  evaluate_rpeaks.py   benchmark against reference annotations
tests/                 32 tests, synthetic signals, no network needed
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .
```

## Use

```bash
python scripts/fetch_data.py                     # cache the benchmark records
python scripts/run_pipeline.py --record 106      # full pipeline + figures
python scripts/evaluate_rpeaks.py                # benchmark
python scripts/evaluate_rpeaks.py --all          # all 48 MIT-BIH records
python -m pytest -q                              # tests
```

## Notes

PhysioNet returns intermittent 502s; `fetch_data.py` retries with backoff and
caches locally so benchmarks stay reproducible offline.

If TLS interception is active on the machine (corporate proxy or antivirus),
PhysioNet downloads fail certificate verification. Generate a bundle combining
the system roots with the intercepting root at `configs/ca-bundle.pem`; the
loader picks it up automatically. The bundle is machine-specific and is not
committed.

## Next

1. Beat-morphology features and PVC classification against the 520 annotated
   ventricular beats in record 106 (weeks 3–4).
2. A position-independent detector, so cross-position comparisons measure
   physiology rather than detector bias.
3. Long-duration analytics: hourly and 24-hour summaries, arrhythmia burden,
   temporal distribution.
4. Backend and dashboard (weeks 7–8).
5. Real sensor integration, then genuine multi-sensor synchronisation, where
   `clock_offset_s` stops being zero and drift estimation begins.
