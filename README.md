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

## Ventricular beat classification

Beat morphology and a logistic-regression classifier, evaluated **inter-patient**:
trained on one group of subjects (de Chazal DS1) and tested on a completely
different group (DS2), with the four paced records excluded per AAMI.

| Protocol | Sensitivity | PPV | F1 |
|---|---|---|---|
| **inter-patient (DS2)** | **92.13%** | **77.25%** | **84.04%** |
| intra-patient | 92.13% | 78.64% | 84.85% |

24,853 test beats, 1,614 ventricular, identical beats in both rows.

Two deliberate choices about the protocol:

**The split holds subjects out whole.** A random split over beats puts the
same person's heartbeats in train and test; because one subject's beats all
look alike, a model can then recognise the *subject* instead of the pathology
and report a number that collapses on a new patient. The intra-patient row is
computed only so the size of that effect is visible -- here 0.82 F1 points,
which is small precisely because the features are subject-relative.

**The decision threshold is chosen on training subjects**, via subject-wise
folds inside DS1. Choosing it on the test set would reintroduce the same leak
the split exists to prevent.

Accuracy is not reported. Ventricular beats are a small minority, so a
classifier that answers "normal" to everything scores above 90%.

### Rhythm context did not survive inter-patient evaluation

Prematurity and compensatory pause are textbook PVC indicators, and they
separate clearly *within* a subject. Added to the model they made it worse:
**84.04 F1 with morphology alone against 79.66 with rhythm context added**, on
identical test beats.

Baseline rhythm varies too much between people, and in the atrial-fibrillation
records prematurity is meaningless because every interval is irregular, so
thresholds learned on one group do not carry to another. They bought
sensitivity and cost more precision than they returned.

This is a limitation to revisit rather than a closed question: a more robust
per-subject normalisation may recover the signal, and rhythm context is
expected to be *essential* for supraventricular (PAC) beats, whose morphology
is near-normal and which are separable mainly by timing.

Precision is the weak point -- 438 false positives against 1,487 true ones.
Motion artefact is the most likely contributor, and dedicated artefact
detection is the next thing to add.

```bash
python scripts/train_pvc.py              # inter-patient, reference beats
python scripts/train_pvc.py --detected   # end-to-end, detector output
```

---

## Service

```bash
pip install -e .[server]
python scripts/serve.py            # http://127.0.0.1:8000, API docs at /docs
```

Two ingestion routes, because two things will feed it. `POST
/api/records/{id}/analyse` reads a cached PhysioNet record, which is what
exists today. `POST /api/upload` takes raw samples as JSON, which is what a
sensor bridge will post:

```json
{"record_id": "session-1",
 "channels": [
   {"sensor_id": "A", "position": "upper_chest", "fs": 360,
    "t_start": "2026-01-01T12:00:00+00:00", "samples": [0.01, 0.02]},
   {"sensor_id": "B", "position": "left_lateral_chest", "fs": 360,
    "t_start": "2026-01-01T12:00:03+00:00", "clock_offset_s": 0.012,
    "samples": [0.00, 0.01]}]}
```

Multi-sensor uploads are accepted from the start, each channel carrying its
own sampling rate, start time and clock offset. A three-second stagger
between two sensors produces a recording three seconds longer than either,
rather than two traces falsely aligned at zero.

Both routes converge on one `analyse_recording`, shared with the command
line, so the API and the scripts cannot drift into reporting different
numbers for the same recording. The classifier is fitted once and cached; if
its training records are not present the service says so and returns
detection without classification rather than downloading 22 records mid-request.

`GET /api/records/{id}/report` returns the full dashboard as HTML.

---

## Dashboard

`scripts/build_dashboard.py` runs the whole chain on a recording and writes a
**single self-contained HTML file**: the plotting library is inlined, so the
report opens offline, survives being emailed, and needs no server.

```bash
python scripts/build_dashboard.py --record 106
```

It shows coverage and beat counts, heart-rate and RR trends, a signal-quality
strip, the hourly distribution of abnormal beats, representative event
waveforms overlaid for morphology comparison, a per-sensor table, and the
synchronised multi-sensor traces.

Three things it deliberately does *not* do:

**It does not present derived labels as ground truth.** Every report carries
the classifier's operating point, so a burden figure is read next to the fact
that roughly one in four flagged beats is expected to be a false positive.

**It does not quote rates it cannot support.** Rates are per *analysed* hour,
excluding dropout, and a recording shorter than an hour says its per-hour
figure is extrapolated rather than measured. Burden, being a ratio over beats,
is unaffected and stays comparable across recordings of any length.

**It does not draw through gaps.** Missing stretches stay visibly missing in
the trends instead of being bridged by interpolation.

The classifier is trained on other subjects and applied to the reported
record, so a record is never scored by a model fitted on it.

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
    artifact.py           motion-artefact detection (kurtosis, bSQI, instability)
  detection/
    rpeaks.py             Pan-Tompkins with inspectable stages, gap recovery
    streaming.py          chunked detection for long recordings
  analysis/
    rr.py                 RR intervals, HR trend, time-domain HRV, ectopic screen
    morphology.py         beat extraction, per-subject templates, beat features
    beat_classifier.py    ventricular classification, inter-patient split
    multiclass.py         AAMI N/S/V/F classification
    af.py                 irregular-rhythm (AF) detection
    summary.py            burden, hourly distribution, longitudinal summaries
    evaluation.py         EC57 matching, sensitivity/PPV/F1
  viz/plots.py            detection overview, synchronised multi-channel, HR trend
  viz/dashboard.py        self-contained HTML report
  app/pipeline.py         the analysis chain as one callable
  app/server.py           HTTP service: ingestion, analysis, reports
scripts/
  fetch_data.py           cache PhysioNet records locally (retries 502s)
  run_pipeline.py         full chain on one record, writes figures
  evaluate_rpeaks.py      benchmark against reference annotations
  train_pvc.py            train/evaluate ventricular classification
  train_multiclass.py     train/evaluate AAMI five-class classification
  build_dashboard.py      full chain on one record -> HTML report
  serve.py                run the HTTP service
tests/                    181 tests, synthetic signals, no network needed
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
HRV features, beat morphology and per-subject templates, ventricular beat
classification with inter-patient evaluation, EC57 evaluation, visualisation.

Not yet implemented:

1. Motion-artefact detection. Artefacts resemble ectopic beats, and are the
   likely source of much of the classifier's remaining false-positive rate.
2. Supraventricular (PAC) classification, which needs the rhythm-context
   features that ventricular classification did not benefit from.
3. Atrial-fibrillation detection.
4. A position-independent detector, so cross-position comparisons measure
   physiology rather than detector bias.
5. Long-duration analytics: hourly and 24-hour summaries, arrhythmia burden,
   temporal distribution.
6. Backend and interactive dashboard.
7. Real-sensor integration and clock-drift estimation between sensors, where
   `clock_offset_s` stops being zero.

## Data

MIT-BIH Arrhythmia Database, via PhysioNet. Records are downloaded on demand
and are not redistributed here.

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> IEEE Eng in Med and Biol 20(3):45-50 (2001).
>
> Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet.
> Circulation 101(23):e215-e220 (2000).
