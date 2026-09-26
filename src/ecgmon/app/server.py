"""HTTP service for ECG ingestion and analysis.

This is the shape the eventual product needs: sensors deliver data, the
server processes it, and a client asks for results. It is deliberately a
small service rather than a framework -- there is no live sensor yet, and the
useful thing to have ready when one arrives is a defined ingestion contract,
not a half-built application around it.

Two ingestion routes exist because two things will feed it:

* ``/api/records/{id}/analyse`` reads a PhysioNet record already on disk.
  This is what exists today and what the evaluation runs against.
* ``/api/upload`` takes raw samples as JSON from an arbitrary source, which
  is what a sensor bridge will post. Multi-sensor uploads are accepted from
  the start: the payload is a list of channels, each with its own sampling
  rate, start time and position.

Both converge on ``analyse_recording``, so the API cannot drift from the
command line.
"""

# NOTE: no "from __future__ import annotations" here. It turns annotations
# into strings, and FastAPI then cannot resolve a request model defined
# inside create_app() -- it silently falls back to treating the body as a
# query parameter and every POST fails validation with "field required".

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..analysis.summary import summarize_channel
from ..io.record import ChannelRecord, SynchronizedRecording
from ..io.wfdb_loader import load_wfdb_record
from ..viz.dashboard import build_dashboard
from .pipeline import analyse_recording

# Upload guard. A 24-hour single channel at 360 Hz is ~31 million samples;
# this allows a few hours per channel while refusing a payload that would
# exhaust memory during JSON parsing.
MAX_SAMPLES_PER_CHANNEL = 5_000_000
MAX_CHANNELS = 16


def create_app(data_dir="data/raw/mitdb", out_dir="outputs"):
    """Build the FastAPI application.

    Imported lazily so the rest of the package does not depend on FastAPI.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    app = FastAPI(
        title="ECG Monitor",
        description=(
            "Research prototype for portable long-term ECG monitoring. "
            "Not a medical device; not for diagnostic use."
        ),
        version="0.2.0",
    )

    class ChannelUpload(BaseModel):
        sensor_id: str = Field(..., description="Stable id of the physical sensor")
        fs: float = Field(..., gt=0, description="Sampling rate in Hz")
        samples: List[float] = Field(..., description="Signal samples")
        position: str = Field("unknown", description="Anatomical placement")
        units: str = Field("mV", description="Physical units of the samples")
        t_start: Optional[str] = Field(
            None,
            description=(
                "ISO-8601 start time of the first sample. Supplied per channel "
                "because independent sensors do not share a clock."
            ),
        )
        clock_offset_s: float = Field(
            0.0,
            description=(
                "Correction onto the recording's reference clock, for sensors "
                "whose clocks have drifted apart."
            ),
        )

    class UploadRequest(BaseModel):
        record_id: str = Field("upload", description="Identifier for this recording")
        channels: List[ChannelUpload]
        powerline_hz: Optional[float] = Field(
            60.0, description="Mains frequency to notch; null to disable"
        )

    def _to_recording(req) -> SynchronizedRecording:
        if not req.channels:
            raise HTTPException(400, "no channels supplied")
        if len(req.channels) > MAX_CHANNELS:
            raise HTTPException(400, f"at most {MAX_CHANNELS} channels")

        chans = []
        for c in req.channels:
            n = len(c.samples)
            if n == 0:
                raise HTTPException(400, f"channel {c.sensor_id} has no samples")
            if n > MAX_SAMPLES_PER_CHANNEL:
                raise HTTPException(
                    413,
                    f"channel {c.sensor_id} has {n} samples; "
                    f"limit is {MAX_SAMPLES_PER_CHANNEL}",
                )
            arr = np.asarray(c.samples, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                # Non-finite samples are a legitimate way to mark a data gap,
                # so they are kept: segment detection handles them downstream.
                pass

            t_start = datetime(1970, 1, 1, tzinfo=timezone.utc)
            if c.t_start:
                try:
                    t_start = datetime.fromisoformat(c.t_start)
                except ValueError:
                    raise HTTPException(
                        400, f"channel {c.sensor_id}: t_start is not ISO-8601"
                    )

            chans.append(
                ChannelRecord(
                    sensor_id=c.sensor_id,
                    fs=c.fs,
                    signal=arr,
                    position=c.position,
                    t_start=t_start,
                    units=c.units,
                    clock_offset_s=c.clock_offset_s,
                )
            )

        try:
            return SynchronizedRecording(record_id=req.record_id, channels=chans)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    # ------------------------------------------------------------- routes

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "data_dir": str(data_dir.resolve()),
            "data_available": data_dir.exists(),
        }

    @app.get("/api/records")
    def list_records():
        """Records cached locally and ready to analyse."""
        if not data_dir.exists():
            return {"records": []}
        ids = sorted({p.stem for p in data_dir.glob("*.hea")})
        return {"records": ids, "count": len(ids)}

    @app.post("/api/records/{record_id}/analyse")
    def analyse_record(record_id: str, classify: bool = True):
        try:
            rec = load_wfdb_record(record_id, db="mitdb", data_dir=data_dir)
        except Exception as exc:
            raise HTTPException(404, f"record {record_id!r}: {exc}")
        result = analyse_recording(rec, data_dir, classify=classify)
        return JSONResponse(result.to_json())

    @app.post("/api/upload")
    def upload(req: UploadRequest):
        """Ingest raw samples from an arbitrary source and analyse them."""
        rec = _to_recording(req)
        result = analyse_recording(
            rec, data_dir, powerline_hz=req.powerline_hz, classify=True
        )
        return JSONResponse(result.to_json())

    @app.get("/api/records/{record_id}/report", response_class=HTMLResponse)
    def report(record_id: str):
        """Full interactive dashboard for a record."""
        try:
            rec = load_wfdb_record(record_id, db="mitdb", data_dir=data_dir)
        except Exception as exc:
            raise HTTPException(404, f"record {record_id!r}: {exc}")

        result = analyse_recording(rec, data_dir)
        summaries = {sid: c.summary for sid, c in result.channels.items()}
        peaks = {sid: c.peaks for sid, c in result.channels.items()}
        primary = result.primary()

        caveats = [
            "PVC labels come from an automatic classifier scored at 93% "
            "sensitivity and 78% precision on unseen subjects. Roughly one in "
            "four flagged beats is expected to be a false positive.",
            "Irregular-rhythm detection flags irregular supraventricular "
            "rhythm, of which atrial fibrillation is one cause among several; "
            "it is not a diagnosis of AF.",
            "Rates are per analysed hour, excluding signal dropout.",
            "Research prototype on public data. Not validated on sensor "
            "hardware and not for diagnostic use.",
        ]

        path = build_dashboard(
            rec, summaries, peaks,
            event_label="PVC",
            examples=primary.examples,
            out_path=out_dir / f"report_{record_id}.html",
            title=f"ECG monitoring report - record {record_id}",
            subtitle=(
                f"{rec.n_channels} sensor(s) - "
                f"{rec[0].duration_s / 3600:.2f} h - "
                f"analysed in {result.elapsed_s:.1f}s"
            ),
            caveats=caveats,
        )
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML)

    return app


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ECG Monitor</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f7f8fa;color:#1f2a44;
     font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:40px 20px}
h1{font-size:24px;margin:0 0 6px}
.sub{color:#6b7a90;margin-bottom:28px}
.card{background:#fff;border:1px solid rgba(128,128,128,.2);border-radius:8px;
      padding:18px;margin-bottom:16px}
h2{font-size:15px;margin:0 0 12px}
select,button{font:inherit;padding:8px 12px;border-radius:6px;
              border:1px solid rgba(128,128,128,.35);background:#fff;color:inherit}
button{background:#1f2a44;color:#fff;border-color:#1f2a44;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
pre{background:#f0f2f5;padding:12px;border-radius:6px;overflow-x:auto;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.stat{background:#f0f2f5;border-radius:6px;padding:10px 12px}
.stat .l{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#6b7a90}
.stat .v{font-size:18px;font-weight:600}
.warn{background:#fff8e6;border-left:3px solid #c77d0a;padding:10px 14px;
      margin-top:14px;font-size:13px;border-radius:0 6px 6px 0}
a{color:#2a6f97}
@media (prefers-color-scheme:dark){
 body{background:#12151a;color:#e6e9ef}
 .card{background:#1a1e26;border-color:rgba(160,170,190,.18)}
 pre,.stat{background:#12151a}
 select,button{background:#1a1e26;color:#e6e9ef}
 button{background:#e6e9ef;color:#12151a}
 .warn{background:#2a2312}
}
</style></head><body><div class="wrap">
<h1>ECG Monitor</h1>
<div class="sub">Research prototype &middot; not a medical device</div>

<div class="card">
  <h2>Analyse a cached record</h2>
  <select id="rec"></select>
  <button id="go">Analyse</button>
  <button id="rep">Open report</button>
  <div id="out"></div>
</div>

<div class="card">
  <h2>Sensor ingestion</h2>
  <p>POST samples to <code>/api/upload</code>. Channels carry their own
  sampling rate, start time and clock offset, so several independent sensors
  can be submitted together.</p>
  <pre>{
  "record_id": "session-1",
  "channels": [
    {"sensor_id": "A", "position": "upper_chest", "fs": 360,
     "t_start": "2026-01-01T12:00:00+00:00", "samples": [0.01, 0.02, ...]},
    {"sensor_id": "B", "position": "left_lateral_chest", "fs": 360,
     "t_start": "2026-01-01T12:00:00+00:00", "clock_offset_s": 0.012,
     "samples": [0.00, 0.01, ...]}
  ]
}</pre>
  <p>Interactive API docs at <a href="/docs">/docs</a>.</p>
</div>

<script>
const $ = s => document.querySelector(s);
fetch('/api/records').then(r => r.json()).then(d => {
  const sel = $('#rec');
  (d.records || []).forEach(r => {
    const o = document.createElement('option'); o.value = o.textContent = r;
    sel.appendChild(o);
  });
  if (!d.records || !d.records.length) {
    sel.appendChild(new Option('no records cached', ''));
  }
});
$('#go').onclick = async () => {
  const id = $('#rec').value;
  if (!id) return;
  $('#go').disabled = true;
  $('#out').innerHTML = '<p>Analysing…</p>';
  try {
    const r = await fetch(`/api/records/${id}/analyse`, {method: 'POST'});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'failed');
    let html = '';
    for (const [sid, c] of Object.entries(d.channels)) {
      html += `<h3 style="font-size:13px;margin:16px 0 8px">${sid} — ${c.position}</h3>`;
      html += '<div class="grid">';
      const tiles = [
        ['Beats', c.beats], ['Coverage', c.coverage_pct + '%'],
        ['Mean HR', c.mean_hr_bpm + ' bpm'], ['PVCs', c.pvc_count],
        ['PVC burden', c.pvc_burden_pct + '%'],
        ['Irregular rhythm', c.irregular_rhythm_burden_pct + '%'],
        ['SDNN', c.sdnn_ms + ' ms'], ['Artefact beats', c.artifact_beats],
      ];
      tiles.forEach(([l, v]) =>
        html += `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div></div>`);
      html += '</div>';
      if (c.pvc_rate_extrapolated) {
        html += '<div class="warn">Recording is under an hour; the per-hour ' +
                'rate is projected from it, not measured.</div>';
      }
    }
    html += `<p style="color:#6b7a90;font-size:12px">analysed in ${d.elapsed_s}s</p>`;
    (d.warnings || []).forEach(w => html += `<div class="warn">${w}</div>`);
    $('#out').innerHTML = html;
  } catch (e) {
    $('#out').innerHTML = `<div class="warn">${e.message}</div>`;
  }
  $('#go').disabled = false;
};
$('#rep').onclick = () => {
  const id = $('#rec').value;
  if (id) window.open(`/api/records/${id}/report`, '_blank');
};
</script>
</div></body></html>
"""
