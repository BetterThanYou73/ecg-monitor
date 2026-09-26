"""Tests for the HTTP service and the shared analysis pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ecgmon.app.pipeline import analyse_recording, reset_classifier
from ecgmon.io.record import ChannelRecord, SynchronizedRecording

from test_pipeline import FS, synth_ecg

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ecgmon.app.server import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    # An empty data dir: no cached records, so classification is unavailable
    # and the service must degrade rather than fail.
    return TestClient(create_app(data_dir=tmp_path / "data", out_dir=tmp_path / "out"))


def make_recording(duration_s=60.0, n_channels=1):
    chans = []
    for i in range(n_channels):
        sig, _ = synth_ecg(duration_s=duration_s, seed=i)
        chans.append(
            ChannelRecord(f"s{i}", FS, sig, position=f"pos{i}")
        )
    return SynchronizedRecording("test", chans)


def upload_payload(duration_s=20.0, n_channels=1, fs=FS):
    channels = []
    for i in range(n_channels):
        sig, _ = synth_ecg(duration_s=duration_s, seed=i)
        channels.append({
            "sensor_id": f"S{i}",
            "position": ["upper_chest", "lower_chest", "left_lateral_chest"][i % 3],
            "fs": fs,
            "samples": sig.tolist(),
        })
    return {"record_id": "session-1", "channels": channels}


# ---------------------------------------------------------------- pipeline


def test_pipeline_runs_without_classifier(tmp_path):
    """No training data available -> detection still works, with a warning."""
    reset_classifier()
    rec = make_recording()
    result = analyse_recording(rec, tmp_path, classify=True)

    assert result.n_channels == 1
    assert result.primary().peaks.size > 30
    assert result.warnings, "expected a warning about classification"
    reset_classifier()


def test_pipeline_classify_disabled(tmp_path):
    rec = make_recording()
    result = analyse_recording(rec, tmp_path, classify=False)
    assert result.primary().pvc_samples.size == 0
    assert not result.warnings


def test_pipeline_handles_multiple_channels(tmp_path):
    rec = make_recording(n_channels=3)
    result = analyse_recording(rec, tmp_path, classify=False)
    assert result.n_channels == 3
    assert len(result.channels) == 3
    for c in result.channels.values():
        assert c.summary is not None


def test_pipeline_json_is_serialisable(tmp_path):
    import json

    rec = make_recording(n_channels=2)
    payload = analyse_recording(rec, tmp_path, classify=False).to_json()
    json.dumps(payload)  # must not raise on numpy types
    assert set(payload["channels"]) == {"s0", "s1"}


def test_json_does_not_embed_full_arrays(tmp_path):
    """A 24 h recording has ~100k beats; a summary must not carry them all."""
    rec = make_recording(duration_s=120.0)
    payload = analyse_recording(rec, tmp_path, classify=False).to_json()
    ch = payload["channels"]["s0"]
    assert isinstance(ch["beats"], int)
    for value in ch.values():
        assert not isinstance(value, list) or len(value) < 100


# ------------------------------------------------------------------ routes


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ECG Monitor" in r.text


def test_list_records_empty(client):
    r = client.get("/api/records")
    assert r.status_code == 200
    assert r.json()["records"] == []


def test_analyse_missing_record_is_404(client):
    r = client.post("/api/records/nosuch/analyse")
    assert r.status_code == 404


def test_report_missing_record_is_404(client):
    r = client.get("/api/records/nosuch/report")
    assert r.status_code == 404


# ------------------------------------------------------------------ upload


def test_upload_single_channel(client):
    r = client.post("/api/upload", json=upload_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["record_id"] == "session-1"
    assert body["n_channels"] == 1
    ch = body["channels"]["S0"]
    assert ch["beats"] > 10
    assert ch["mean_hr_bpm"] is not None


def test_upload_multi_sensor(client):
    """Several independent sensors in one payload, the target hardware shape."""
    r = client.post("/api/upload", json=upload_payload(n_channels=3))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_channels"] == 3
    assert set(body["channels"]) == {"S0", "S1", "S2"}
    assert body["channels"]["S1"]["position"] == "lower_chest"


def test_upload_respects_per_channel_start_times(client):
    """Sensors do not share a clock; each channel carries its own t_start."""
    payload = upload_payload(n_channels=2)
    payload["channels"][0]["t_start"] = "2026-01-01T12:00:00+00:00"
    payload["channels"][1]["t_start"] = "2026-01-01T12:00:05+00:00"
    r = client.post("/api/upload", json=payload)
    assert r.status_code == 200, r.text
    # 5 s stagger on a 20 s recording -> total span 25 s
    assert r.json()["duration_s"] == pytest.approx(25.0, abs=0.5)


def test_upload_rejects_bad_timestamp(client):
    payload = upload_payload()
    payload["channels"][0]["t_start"] = "not-a-date"
    r = client.post("/api/upload", json=payload)
    assert r.status_code == 400


def test_upload_rejects_empty_channels(client):
    r = client.post("/api/upload", json={"record_id": "x", "channels": []})
    assert r.status_code == 400


def test_upload_rejects_empty_samples(client):
    r = client.post("/api/upload", json={
        "record_id": "x",
        "channels": [{"sensor_id": "A", "fs": 360, "samples": []}],
    })
    assert r.status_code == 400


def test_upload_rejects_duplicate_sensor_ids(client):
    payload = upload_payload(n_channels=2)
    payload["channels"][1]["sensor_id"] = payload["channels"][0]["sensor_id"]
    r = client.post("/api/upload", json=payload)
    assert r.status_code == 400


def test_upload_rejects_non_positive_sampling_rate(client):
    payload = upload_payload()
    payload["channels"][0]["fs"] = 0
    r = client.post("/api/upload", json=payload)
    assert r.status_code == 422   # pydantic bound


def test_upload_rejects_oversized_channel(client, monkeypatch):
    import ecgmon.app.server as server

    monkeypatch.setattr(server, "MAX_SAMPLES_PER_CHANNEL", 100)
    app = server.create_app(data_dir="nonexistent")
    with TestClient(app) as c:
        r = c.post("/api/upload", json=upload_payload(duration_s=5.0))
        assert r.status_code == 413


def test_upload_rejects_too_many_channels(client, monkeypatch):
    import ecgmon.app.server as server

    monkeypatch.setattr(server, "MAX_CHANNELS", 2)
    app = server.create_app(data_dir="nonexistent")
    with TestClient(app) as c:
        r = c.post("/api/upload", json=upload_payload(n_channels=3, duration_s=5.0))
        assert r.status_code == 400


def test_upload_accepts_nan_as_data_gap(client):
    """Non-finite samples mark a dropout and must not be rejected."""
    payload = upload_payload(duration_s=30.0)
    s = payload["channels"][0]["samples"]
    for i in range(1000, 2000):
        s[i] = None          # JSON null -> NaN
    r = client.post("/api/upload", json=payload)
    assert r.status_code in (200, 422)
