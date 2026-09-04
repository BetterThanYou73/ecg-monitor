# ECG-Monitor

A signal-analysis pipeline for long-duration ECG monitoring, built to work
with **multiple independent single-channel sensors recorded synchronously**
as naturally as with one.

Wearable ECG is increasingly recorded by small independent sensors rather
than by a single multi-lead amplifier. That changes the software problem in a
specific way: the channels no longer share a clock. Anything that compares
signals across sensors has to reason about absolute time, per-sensor offsets
and drift, not sample indices.

So the data model here treats a **single-sensor recording as the one-channel
case of a multi-sensor recording**. Per-channel sampling rate, absolute start
time and clock offset are present from the beginning, and the shared-timeline
arithmetic is pinned by tests. Adding sensors later needs no restructuring.

---

## Results

R-peak detection against MIT-BIH Arrhythmia Database reference annotations.
8 records, 18,552 annotated beats, 4.0 hours of ECG. Matching follows
ANSI/AAMI EC57: 150 ms tolerance, each reference beat matched at most once.

| Detector | Lead | Sensitivity | PPV | F1 |
|---|---|---|---|---|
| Pan-Tompkins (this repo) | 0 (mostly MLII) | 98.37% | 99.27% | **98.82%** |
| Pan-Tompkins + gap recovery | 0 | 99.50% | 97.96% | 98.72% |
| NeuroKit2 (reference) | 0 | 97.48% | 98.73% | 98.10% |
| Pan-Tompkins | 1 (V1 etc.) | 83.96% | 97.27% | 90.13% |
| Pan-Tompkins + gap recovery | 1 | 89.36% | 94.89% | 92.04% |

Throughput is roughly 26,000x realtime: a 24-hour single-channel recording
processes in about 3 seconds.

Sensitivity and PPV are reported rather than accuracy. Beats vastly outnumber
non-beats, so accuracy stays high even when a clinically significant fraction
of beats is missed.

### Detection accuracy is strongly position-dependent

The same detector, on the same beats in the same recordings, scores 98.4%
sensitivity on lead 0 and 84.0% on lead 1.

The cause is diagnosed rather than assumed. In records with frequent ectopic
beats, the secondary lead shows large ectopic beats alongside small normal
ones. Pan-Tompkins tracks a single running signal-peak estimate, so the large
beats pull the threshold above the small ones and normal beats are lost in
runs. On record 106 lead V1 this costs 62 percentage points of sensitivity.

This matters for any multi-sensor comparison: some of the apparent difference
between recording positions is detector bias rather than physiology, and the
two have to be separated before drawing conclusions about which positions
detect what. A position-independent detector is a prerequisite for sensor
fusion work, not an optimisation of it.

`--recover` enables a second pass that re-scans long gaps at a locally
derived threshold. It trades PPV for sensitivity and is off by default,
because inventing beats inside a genuine pause corrupts the rhythm analysis
that the pause feeds into.

---

## Long-recording handling

A 24-hour ambulatory recording is not 24 hours of usable ECG, and does not
fit comfortably in memory.

**Dropouts.** Electrodes detach, subjects move, batteries interrupt, wireless
sensors drop packets. Analysing straight through produces confident nonsense:
a flatline yields no beats, which reads downstream as a long pause, which
reads as an arrhythmia. `preprocessing.segments` finds non-finite gaps and
sustained flatlines, analysis runs only on usable stretches, and coverage is
reported alongside every result so a burden figure is always relative to how
much signal actually existed.

**Memory.** Detecting on a whole 24-hour array means holding the signal plus
four Pan-Tompkins intermediate arrays at once — roughly 1.2 GB at 360 Hz in
float64. `detection.streaming` processes overlapping chunks, bounding memory
regardless of recording length. Chunks overlap so that a beat on a seam is
neither missed by both chunks nor counted twice, and so the adaptive
thresholds get a run-up before the region whose detections are kept.
Chunked output agrees with whole-signal detection at F1 > 99.3% across the
benchmark set.

---

## Layout

```
src/ecgmon/
  io/record.py            data model: ChannelRecord, SynchronizedRecording
  io/wfdb_loader.py       PhysioNet/WFDB ingestion and beat annotations
  preprocessing/
    filters.py            baseline wander, mains notch, bandpass, resampling
    quality.py            windowed signal-quality index, sensor ranking
    segments.py           dropout and gap detection, coverage reporting
  detection/
    rpeaks.py             Pan-Tompkins with inspectable stages, gap recovery
    streaming.py          chunked detection for long recordings
  analysis/
    rr.py                 RR intervals, HR trend, time-domain HRV, ectopic screen
    evaluation.py         EC57 matching, sensitivity/PPV/F1
  viz/plots.py            detection overview, synchronised multi-channel, HR trend
scripts/
  fetch_data.py           cache PhysioNet records locally (retries 502s)
  run_pipeline.py         full chain on one record, writes figures
  evaluate_rpeaks.py      benchmark against reference annotations
tests/                    46 tests, synthetic signals, no network needed
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

PhysioNet returns intermittent 502s under load; `fetch_data.py` retries with
backoff and caches locally so benchmarks stay reproducible offline.

If TLS interception is active on the machine (corporate proxy or antivirus),
PhysioNet downloads fail certificate verification. Generate a bundle combining
the system roots with the intercepting root at `configs/ca-bundle.pem`; the
loader picks it up automatically. The bundle is machine-specific and is not
committed.

## Status and next steps

Implemented: ingestion, preprocessing, signal-quality assessment, dropout and
coverage handling, R-peak detection, chunked long-recording processing, RR/HR/
HRV features, a rhythm-level ectopic screen, EC57 evaluation, visualisation.

Not yet implemented:

1. Beat-morphology features and PVC classification. The ectopic screen is
   rhythm-only — it narrows the search, it does not diagnose. Record 106
   carries 520 annotated ventricular beats as a validation target.
2. A position-independent detector, so cross-position comparisons measure
   physiology rather than detector bias.
3. Long-duration analytics: hourly and 24-hour summaries, arrhythmia burden,
   temporal distribution.
4. Backend and interactive dashboard.
5. Real-sensor integration and clock-drift estimation between sensors, where
   `clock_offset_s` stops being zero.

## Data

MIT-BIH Arrhythmia Database, via PhysioNet. Records are downloaded on demand
and are not redistributed here.

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> IEEE Eng in Med and Biol 20(3):45-50 (2001).
>
> Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet.
> Circulation 101(23):e215-e220 (2000).
