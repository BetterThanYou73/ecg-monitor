"""Interactive monitoring dashboard.

Produces a single self-contained HTML file: Plotly is inlined, so the report
opens offline, survives being emailed, and needs no server. That matters
while there is no live data source -- a report a supervisor can double-click
is worth more right now than a web service with nothing plugged into it.

The layout is driven by what a reviewer asks of a 24-hour recording, roughly
in order: how much of this is trustworthy, what happened overall, when did it
happen, and what did it actually look like.

Built multi-sensor first, like the rest of the pipeline. With one channel the
per-sensor sections collapse to a single column; with several they sit side
by side on a shared timeline.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

# Palette chosen to stay legible in both light and dark viewers, and to keep
# the abnormal-event colour distinct from the signal trace at a glance.
COL_BG = "#ffffff"
COL_INK = "#1f2a44"
COL_MUTED = "#6b7a90"
COL_LINE = "#2a6f97"
COL_EVENT = "#d1495b"
COL_GOOD = "#3a7d44"
COL_WARN = "#c77d0a"


def _plotly():
    import plotly.graph_objects as go

    return go


def _fig_html(fig, include_js: bool) -> str:
    """Render one figure, inlining plotly.js only on the first call."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"displaylogo": False, "responsive": True},
    )


def _layout(fig, title: str, height: int = 280, ytitle: str = "", xtitle: str = ""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COL_INK)),
        height=height,
        margin=dict(l=60, r=20, t=44, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COL_INK, size=12),
        xaxis=dict(title=xtitle, gridcolor="rgba(128,128,128,0.18)", zeroline=False),
        yaxis=dict(title=ytitle, gridcolor="rgba(128,128,128,0.18)", zeroline=False),
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def _stat(label: str, value: str, note: str = "", tone: str = "") -> str:
    colour = {"good": COL_GOOD, "warn": COL_WARN, "event": COL_EVENT}.get(tone, COL_INK)
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="stat"><div class="label">{html.escape(label)}</div>'
        f'<div class="value" style="color:{colour}">{html.escape(value)}</div>'
        f"{note_html}</div>"
    )


def _hr_figure(summary, include_js: bool) -> str:
    go = _plotly()
    fig = go.Figure()
    if summary.hr_trend_t.size:
        fig.add_trace(
            go.Scatter(
                x=summary.hr_trend_t / 60.0,
                y=summary.hr_trend_bpm,
                mode="lines",
                line=dict(color=COL_LINE, width=1.6),
                name="HR",
                connectgaps=False,  # gaps are data, not a drawing artefact
                hovertemplate="%{y:.0f} bpm at %{x:.1f} min<extra></extra>",
            )
        )
    _layout(fig, "Heart rate", ytitle="bpm", xtitle="minutes")
    return _fig_html(fig, include_js)


def _events_figure(summary, label: str, include_js: bool) -> str:
    """Hourly distribution of one abnormal-beat class."""
    go = _plotly()
    ev = summary.events.get(label)
    fig = go.Figure()
    if ev is not None and ev.hourly_counts.size:
        centres = ev.hour_edges[:-1] + 0.5
        fig.add_trace(
            go.Bar(
                x=centres,
                y=ev.hourly_counts,
                marker_color=COL_EVENT,
                hovertemplate="hour %{x:.0f}: %{y} events<extra></extra>",
            )
        )
    _layout(fig, f"{label} events per hour", ytitle="count", xtitle="hour of recording")
    return _fig_html(fig, include_js)


def _rr_figure(summary, include_js: bool) -> str:
    go = _plotly()
    fig = go.Figure()
    if summary.rr_t.size:
        # Subsample for very long recordings: a 24 h trace is ~100k points and
        # browsers stop being responsive well before that.
        step = max(1, summary.rr_t.size // 8000)
        fig.add_trace(
            go.Scattergl(
                x=summary.rr_t[::step] / 60.0,
                y=summary.rr_s[::step] * 1000.0,
                mode="markers",
                marker=dict(size=2.5, color=COL_LINE, opacity=0.55),
                hovertemplate="%{y:.0f} ms at %{x:.1f} min<extra></extra>",
            )
        )
    _layout(fig, "RR intervals", ytitle="ms", xtitle="minutes")
    return _fig_html(fig, include_js)


def _sqi_figure(summary, include_js: bool) -> str:
    go = _plotly()
    fig = go.Figure()
    if summary.sqi_t.size:
        fig.add_trace(
            go.Scatter(
                x=summary.sqi_t / 60.0,
                y=summary.sqi,
                mode="lines",
                line=dict(color=COL_GOOD, width=1.2),
                fill="tozeroy",
                fillcolor="rgba(58,125,68,0.15)",
                hovertemplate="SQI %{y:.2f} at %{x:.1f} min<extra></extra>",
            )
        )
        fig.add_hline(y=0.5, line=dict(color=COL_WARN, width=1, dash="dash"))
    _layout(fig, "Signal quality", ytitle="SQI", xtitle="minutes")
    fig.update_yaxes(range=[0, 1])
    return _fig_html(fig, include_js)


def _waveforms_figure(examples: list, label: str, include_js: bool) -> str:
    """Representative event waveforms, drawn on a shared time axis."""
    go = _plotly()
    fig = go.Figure()
    for i, (centre, t, wave) in enumerate(examples):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=wave + i * 0.0,  # no offset: overlay so morphology compares
                mode="lines",
                line=dict(width=1.1),
                opacity=0.75,
                name=f"t={centre}",
                hovertemplate=f"event at sample {centre}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line=dict(color=COL_MUTED, width=1, dash="dot"))
    _layout(fig, f"Representative {label} waveforms (overlaid, aligned on beat)",
            ytitle="mV", xtitle="seconds from event")
    return _fig_html(fig, include_js)


def _multichannel_figure(recording, peaks_by_sensor, window_s, include_js: bool) -> str:
    """Synchronised traces from every sensor on one reference timeline."""
    go = _plotly()
    from plotly.subplots import make_subplots

    n = recording.n_channels
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[f"{c.sensor_id} — {c.position}" for c in recording])
    origin = recording.t_origin
    t0, t1 = window_s

    for r, ch in enumerate(recording, start=1):
        elapsed = ch.elapsed_since(origin)
        m = (elapsed >= t0) & (elapsed <= t1)
        fig.add_trace(
            go.Scattergl(x=elapsed[m], y=ch.signal[m], mode="lines",
                         line=dict(color=COL_INK, width=1)),
            row=r, col=1,
        )
        pk = np.asarray(peaks_by_sensor.get(ch.sensor_id, []), dtype=int)
        pk = pk[(pk >= 0) & (pk < ch.n_samples)]
        if pk.size:
            pt = elapsed[pk]
            sel = (pt >= t0) & (pt <= t1)
            fig.add_trace(
                go.Scattergl(x=pt[sel], y=ch.signal[pk][sel], mode="markers",
                             marker=dict(color=COL_EVENT, size=6, symbol="triangle-down")),
                row=r, col=1,
            )

    fig.update_layout(
        height=190 * n + 60,
        margin=dict(l=60, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COL_INK, size=11), showlegend=False,
        title=dict(text="Synchronised sensor traces", font=dict(size=14)),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)",
                     title_text="seconds from recording origin", row=n, col=1)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    for ann in fig.layout.annotations:
        ann.font.size = 11
    return _fig_html(fig, include_js)


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f7f8fa;color:#1f2a44;
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
header{border-bottom:2px solid #1f2a44;padding-bottom:14px;margin-bottom:22px}
h1{margin:0 0 4px;font-size:22px;letter-spacing:-.01em}
.sub{color:#6b7a90;font-size:13px}
h2{font-size:15px;margin:30px 0 12px;padding-bottom:6px;
   border-bottom:1px solid rgba(128,128,128,.25);letter-spacing:.01em}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:14px 0 6px}
.stat{background:#fff;border:1px solid rgba(128,128,128,.2);border-radius:8px;padding:12px 14px}
.stat .label{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#6b7a90}
.stat .value{font-size:21px;font-weight:600;margin-top:3px}
.stat .note{font-size:11px;color:#6b7a90;margin-top:2px}
.card{background:#fff;border:1px solid rgba(128,128,128,.2);border-radius:8px;
      padding:6px 10px;margin-bottom:16px;overflow-x:auto}
.caveat{background:#fff8e6;border-left:3px solid #c77d0a;padding:10px 14px;
        border-radius:0 6px 6px 0;margin:14px 0;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid rgba(128,128,128,.18)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#6b7a90}
td.num{text-align:right;font-variant-numeric:tabular-nums}
footer{margin-top:36px;padding-top:14px;border-top:1px solid rgba(128,128,128,.25);
       color:#6b7a90;font-size:12px}
@media (prefers-color-scheme:dark){
  body{background:#12151a;color:#e6e9ef}
  .stat,.card,table{background:#1a1e26;border-color:rgba(160,170,190,.18)}
  header{border-bottom-color:#e6e9ef}
  .caveat{background:#2a2312;border-left-color:#c77d0a}
  th,.sub,.stat .label,.stat .note,footer{color:#9aa6b8}
}
"""


def build_dashboard(
    recording,
    summaries: dict,
    peaks_by_sensor: dict,
    event_label: str = "PVC",
    examples: list | None = None,
    out_path: str | Path = "outputs/dashboard.html",
    title: str = "ECG monitoring report",
    subtitle: str = "",
    caveats: list | None = None,
    trace_window_s: tuple = (0.0, 10.0),
    inline_plotly: bool = True,
) -> Path:
    """Render the full dashboard to a self-contained HTML file.

    Args:
        recording: ``SynchronizedRecording`` being reported on.
        summaries: ``{sensor_id: ChannelSummary}``.
        peaks_by_sensor: ``{sensor_id: peak indices}``.
        event_label: Name of the abnormal-beat class being summarised.
        examples: Output of ``representative_beats`` for the primary sensor.
        caveats: Statements of what the numbers do not mean. Rendered
            prominently rather than in a footnote -- a dashboard that shows an
            arrhythmia burden without saying how it was derived invites the
            reader to trust it further than the evidence allows.
        inline_plotly: Embed the plotting library (~3.5 MB) so the file opens
            offline. Always true for real reports; tests turn it off, since
            embedding it repeatedly makes the suite too slow to be a useful
            feedback loop.
    """
    parts: list[str] = []
    first = inline_plotly  # only the first figure carries the inlined plotly.js

    primary_id = next(iter(summaries))
    primary = summaries[primary_id]

    # ------------------------------------------------------------- header
    total_beats = sum(s.n_beats for s in summaries.values())
    ev = primary.events.get(event_label)

    parts.append('<div class="wrap"><header>')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    sub = subtitle or (
        f"{recording.record_id} · {recording.n_channels} sensor(s) · "
        f"{primary.duration_s / 3600:.2f} h"
    )
    parts.append(f'<div class="sub">{html.escape(sub)}</div></header>')

    # -------------------------------------------------------- key figures
    parts.append("<h2>Overview</h2>")
    parts.append('<div class="stats">')
    parts.append(_stat("Recording", f"{primary.duration_s / 3600:.2f} h"))
    parts.append(
        _stat("Analysed", f"{primary.coverage.fraction * 100:.1f}%",
              f"{primary.analysed_hours:.2f} h usable, {primary.coverage.n_gaps} gap(s)",
              tone="good" if primary.coverage.fraction > 0.9 else "warn")
    )
    parts.append(_stat("Beats", f"{total_beats:,}"))
    parts.append(_stat("Mean HR", f"{primary.mean_hr:.0f} bpm",
                       f"min {primary.min_hr:.0f} / max {primary.max_hr:.0f}"))
    if ev is not None:
        parts.append(_stat(f"{event_label} count", f"{ev.count:,}", tone="event"))
        rate_note = f"{ev.per_hour:.1f}/h, peak hour {ev.max_hourly}"
        if ev.rate_is_extrapolated:
            rate_note = (f"{ev.per_hour:.0f}/h extrapolated from "
                         f"{ev.analysed_hours:.2f} h")
        parts.append(_stat(f"{event_label} burden", f"{ev.burden_pct:.2f}%",
                           rate_note, tone="event"))
    parts.append(_stat("SDNN", f"{primary.sdnn_ms:.0f} ms"))
    parts.append(_stat("RMSSD", f"{primary.rmssd_ms:.0f} ms"))
    parts.append("</div>")

    if ev is not None and ev.rate_is_extrapolated:
        caveats = list(caveats or []) + [
            f"This recording is {ev.analysed_hours:.2f} h long. The per-hour "
            f"rate is projected from it, not measured over an hour; the burden "
            f"percentage is unaffected."
        ]

    if caveats:
        parts.append('<div class="caveat"><strong>How to read this.</strong><ul>')
        for c in caveats:
            parts.append(f"<li>{html.escape(c)}</li>")
        parts.append("</ul></div>")

    # ------------------------------------------------------------- trends
    parts.append("<h2>Trends</h2>")
    for fig_html in (
        _hr_figure(primary, first),
        _rr_figure(primary, False),
        _sqi_figure(primary, False),
    ):
        parts.append(f'<div class="card">{fig_html}</div>')
        first = False

    # ------------------------------------------------------------- events
    if ev is not None:
        parts.append(f"<h2>{html.escape(event_label)} distribution</h2>")
        parts.append(f'<div class="card">{_events_figure(primary, event_label, False)}</div>')
        if examples:
            parts.append(
                f'<div class="card">{_waveforms_figure(examples, event_label, False)}</div>'
            )

    # ------------------------------------------------------------ sensors
    parts.append("<h2>Per-sensor comparison</h2>")
    parts.append("<table><tr><th>Sensor</th><th>Position</th><th>Coverage</th>"
                 "<th>Beats</th><th>Mean HR</th>"
                 f"<th>{html.escape(event_label)}</th><th>Burden</th></tr>")
    for sid, s in summaries.items():
        e = s.events.get(event_label)
        parts.append(
            f"<tr><td>{html.escape(sid)}</td><td>{html.escape(s.position)}</td>"
            f'<td class="num">{s.coverage.fraction * 100:.1f}%</td>'
            f'<td class="num">{s.n_beats:,}</td>'
            f'<td class="num">{s.mean_hr:.0f}</td>'
            f'<td class="num">{e.count if e else 0:,}</td>'
            f'<td class="num">{e.burden_pct if e else 0:.2f}%</td></tr>'
        )
    parts.append("</table>")

    parts.append(
        f'<div class="card">'
        f"{_multichannel_figure(recording, peaks_by_sensor, trace_window_s, False)}</div>"
    )

    parts.append(
        '<footer>Generated by ecgmon. Research prototype — not a medical '
        "device and not for diagnostic use.</footer></div>"
    )

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        + "".join(parts)
        + "</body></html>"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
