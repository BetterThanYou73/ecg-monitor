"""The analysis pipeline as a single callable, for the service to invoke.

The scripts each wire the stages together their own way, which is fine for
experiments but means the API and the command line can drift apart and report
different numbers for the same recording. This module is the one definition
of "process a recording", and both call it.

Everything expensive that does not depend on the recording -- fitting the
beat classifier -- happens once and is cached, because a service that refits
a model on every request spends all its time doing that.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.af import af_burden, af_episodes, detect_af
from ..analysis.beat_classifier import DS1_TRAIN, VentricularClassifier
from ..analysis.morphology import (
    beat_features,
    classifier_columns,
    features_to_array,
)
from ..analysis.summary import representative_beats, summarize_channel
from ..detection.rpeaks import detect_rpeaks
from ..io.record import SynchronizedRecording
from ..preprocessing.artifact import beats_in_artifact
from ..preprocessing.filters import preprocess_channel
from ..preprocessing.quality import assess_channel

_MODEL_LOCK = threading.Lock()
_MODEL: tuple | None = None


@dataclass
class ChannelResult:
    """Per-sensor analysis output."""

    sensor_id: str
    position: str
    peaks: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    pvc_samples: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    artifact_beats: int = 0
    af_windows: list = field(repr=False, default_factory=list)
    af_burden: float = 0.0
    af_episodes: list = field(default_factory=list)
    summary: object = None
    examples: list = field(repr=False, default_factory=list)


@dataclass
class AnalysisResult:
    """Everything produced for one recording."""

    record_id: str
    n_channels: int
    duration_s: float
    channels: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    warnings: list = field(default_factory=list)

    def primary(self) -> ChannelResult:
        return next(iter(self.channels.values()))

    def to_json(self) -> dict:
        """Plain-data view for an HTTP response.

        Arrays are reduced to counts and summary statistics rather than being
        serialised whole: a 24-hour recording holds around 100,000 beats, and
        a caller asking for a summary does not want them all in the payload.
        """
        out = {
            "record_id": self.record_id,
            "n_channels": self.n_channels,
            "duration_s": round(self.duration_s, 2),
            "elapsed_s": round(self.elapsed_s, 2),
            "warnings": list(self.warnings),
            "channels": {},
        }
        for sid, c in self.channels.items():
            s = c.summary
            ev = s.events.get("PVC") if s else None
            out["channels"][sid] = {
                "position": c.position,
                "beats": int(c.peaks.size),
                "coverage_pct": round(s.coverage.fraction * 100, 2) if s else None,
                "coverage_gaps": s.coverage.n_gaps if s else None,
                "mean_hr_bpm": round(s.mean_hr, 1) if s else None,
                "min_hr_bpm": round(s.min_hr, 1) if s else None,
                "max_hr_bpm": round(s.max_hr, 1) if s else None,
                "sdnn_ms": round(s.sdnn_ms, 1) if s else None,
                "rmssd_ms": round(s.rmssd_ms, 1) if s else None,
                "pvc_count": int(c.pvc_samples.size),
                "pvc_burden_pct": round(ev.burden_pct, 2) if ev else 0.0,
                "pvc_per_hour": round(ev.per_hour, 1) if ev else 0.0,
                "pvc_rate_extrapolated": bool(ev.rate_is_extrapolated) if ev else False,
                "artifact_beats": int(c.artifact_beats),
                "irregular_rhythm_burden_pct": round(c.af_burden * 100, 2),
                "irregular_rhythm_episodes": [
                    {"start_s": round(a, 1), "end_s": round(b, 1)}
                    for a, b in c.af_episodes
                ],
            }
        return out


def get_classifier(data_dir: Path) -> tuple:
    """Fit the ventricular classifier once and reuse it.

    Guarded by a lock: two requests arriving together would otherwise each
    start a fit, and the fit reads every training record from disk.
    """
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            # Refuse rather than fetch. Without this the training set is
            # downloaded from PhysioNet on first request, turning a service
            # start into minutes of network traffic and making tests that
            # point at an empty directory pull 22 records.
            data_dir = Path(data_dir)
            cached = sorted(p.stem for p in data_dir.glob("*.hea")) if data_dir.exists() else []
            available = [r for r in DS1_TRAIN if r in cached]
            if len(available) < 5:
                raise FileNotFoundError(
                    f"only {len(available)} of {len(DS1_TRAIN)} training records "
                    f"are cached in {data_dir}; run scripts/fetch_data.py --all"
                )

            import sys

            scripts = Path(__file__).resolve().parents[3] / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            from train_pvc import build_set, select_threshold

            import io
            import contextlib

            # build_set narrates per-record progress, which is noise in a service.
            with contextlib.redirect_stdout(io.StringIO()):
                X, y, g, _ = build_set(available, data_dir, use_detected=False)
            threshold = select_threshold(X, y, g)
            _MODEL = (VentricularClassifier().fit(X, y), threshold)
        return _MODEL


def reset_classifier() -> None:
    """Drop the cached model. Used by tests."""
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None


def analyse_recording(
    recording: SynchronizedRecording,
    data_dir: Path,
    powerline_hz: float | None = 60.0,
    classify: bool = True,
    n_examples: int = 6,
) -> AnalysisResult:
    """Run the full chain over every channel of a recording."""
    t0 = time.perf_counter()
    result = AnalysisResult(
        record_id=recording.record_id,
        n_channels=recording.n_channels,
        duration_s=recording.duration_s,
    )

    clf = threshold = None
    if classify:
        try:
            clf, threshold = get_classifier(data_dir)
        except (Exception, SystemExit) as exc:
            result.warnings.append(
                f"beat classification unavailable ({type(exc).__name__}); "
                "reporting detection only"
            )

    for i, raw in enumerate(recording):
        clean = preprocess_channel(raw, powerline_hz=powerline_hz)
        peaks = detect_rpeaks(clean.signal, clean.fs)

        pvc = np.array([], dtype=int)
        if clf is not None and peaks.size:
            feats, _, _ = beat_features(clean.signal, clean.fs, peaks)
            if feats:
                X = features_to_array(feats)[:, classifier_columns()]
                pred = clf.predict(X, threshold=threshold)
                pvc = np.array(
                    [f.sample for f, flag in zip(feats, pred) if flag], dtype=int
                )

        artifact = beats_in_artifact(peaks, raw) if peaks.size else np.array([], bool)

        # Ectopic beats are excluded before judging rhythm: their irregularity
        # is otherwise indistinguishable from fibrillation.
        ectopic = np.isin(peaks, pvc) if peaks.size else np.array([], bool)
        af_wins = detect_af(peaks, clean.fs, exclude=ectopic) if peaks.size else []

        sqi = assess_channel(raw, window_s=10.0)
        summary = summarize_channel(
            raw, peaks, event_samples={"PVC": pvc}, sqi_windows=sqi
        )

        result.channels[raw.sensor_id] = ChannelResult(
            sensor_id=raw.sensor_id,
            position=raw.position,
            peaks=peaks,
            pvc_samples=pvc,
            artifact_beats=int(np.count_nonzero(artifact)),
            af_windows=af_wins,
            af_burden=af_burden(af_wins),
            af_episodes=af_episodes(af_wins),
            summary=summary,
            examples=(
                representative_beats(clean.signal, clean.fs, pvc, n=n_examples)
                if i == 0
                else []
            ),
        )

    result.elapsed_s = time.perf_counter() - t0
    return result
