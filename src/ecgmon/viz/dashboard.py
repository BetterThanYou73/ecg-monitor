"""Interactive monitoring dashboard.

Produces a single self-contained HTML file: the plotting library is inlined,
so the report opens offline, survives being emailed, and needs no server.
While there is no live data source, a report that can be double-clicked is
worth more than a service with nothing plugged into it.

Two things this file has to get right that a chart library does not do for it.

**Theme.** The page follows the viewer's light or dark setting, but a plot's
axis text, tick labels and gridlines are baked in when the figure is
serialised. Fixed colours mean one theme renders dark text on a dark ground.
So the palette is applied at view time: the page reads the effective theme,
restyles every plot through it, and listens for the setting changing.

**Explaining itself.** Each chart carries a sentence saying what it shows and
how to read it. A burden figure or an RR scatter is not self-evident to
someone seeing the system for the first time, and a dashboard that needs its
author standing next to it is not finished.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

# Semantic roles, resolved to light/dark values at view time by interactive.js.
# Kept as tokens here so the Python and the JavaScript cannot disagree.
ROLE_INK = "ink"
ROLE_MUTED = "muted"
ROLE_GRID = "grid"
ROLE_TRACE = "trace"
ROLE_EVENT = "event"
ROLE_GOOD = "good"

# Placeholder colours used when the figure is serialised. Every one of them is
# overwritten by the theme script before the viewer sees it; they exist only so
# the JSON is valid and so a plot is not invisible if scripting is blocked.
_PLACEHOLDER = {
    ROLE_INK: "#8a94a3",
    ROLE_MUTED: "#8a94a3",
    ROLE_GRID: "rgba(138,148,163,0.22)",
    ROLE_TRACE: "#3d8fb0",
    ROLE_EVENT: "#d1495b",
    ROLE_GOOD: "#4f9e72",
}


def _plotly():
    import plotly.graph_objects as go

    return go


def _fig_html(fig, include_js: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={"displaylogo": False, "responsive": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]},
    )


def _layout(fig, height=260, ytitle="", xtitle=""):
    """Chart chrome. Titles live in the page, not in the plot.

    A plot-level title duplicates the heading already above it and eats
    vertical space the data could use.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=58, r=16, t=10, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_PLACEHOLDER[ROLE_MUTED], size=11.5,
                  family='ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif'),
        xaxis=dict(title=dict(text=xtitle, font=dict(size=11)),
                   gridcolor=_PLACEHOLDER[ROLE_GRID], zeroline=False,
                   linecolor=_PLACEHOLDER[ROLE_GRID]),
        yaxis=dict(title=dict(text=ytitle, font=dict(size=11)),
                   gridcolor=_PLACEHOLDER[ROLE_GRID], zeroline=False,
                   linecolor=_PLACEHOLDER[ROLE_GRID]),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(font_size=12),
    )
    return fig


# --------------------------------------------------------------- figures


def _hr_figure(summary, include_js):
    go = _plotly()
    fig = go.Figure()
    if summary.hr_trend_t.size:
        fig.add_trace(go.Scatter(
            x=summary.hr_trend_t / 60.0, y=summary.hr_trend_bpm, mode="lines",
            line=dict(color=_PLACEHOLDER[ROLE_TRACE], width=2, shape="spline"),
            connectgaps=False,   # a gap in the data is information, not a glitch
            hovertemplate="%{y:.0f} bpm<extra></extra>"))
    _layout(fig, ytitle="bpm", xtitle="minutes into recording")
    return _fig_html(fig, include_js)


def _rr_figure(summary, include_js):
    go = _plotly()
    fig = go.Figure()
    if summary.rr_t.size:
        # A 24 h trace is ~100k points; browsers stall well before that.
        step = max(1, summary.rr_t.size // 6000)
        fig.add_trace(go.Scattergl(
            x=summary.rr_t[::step] / 60.0, y=summary.rr_s[::step] * 1000.0,
            mode="markers",
            marker=dict(size=3, color=_PLACEHOLDER[ROLE_TRACE], opacity=.5),
            hovertemplate="%{y:.0f} ms<extra></extra>"))
    _layout(fig, ytitle="interval (ms)", xtitle="minutes into recording")
    return _fig_html(fig, include_js)


def _sqi_figure(summary, include_js):
    go = _plotly()
    fig = go.Figure()
    if summary.sqi_t.size:
        fig.add_trace(go.Scatter(
            x=summary.sqi_t / 60.0, y=summary.sqi, mode="lines",
            line=dict(color=_PLACEHOLDER[ROLE_GOOD], width=1.6),
            fill="tozeroy", fillcolor="rgba(79,158,114,0.16)",
            hovertemplate="SQI %{y:.2f}<extra></extra>"))
        fig.add_hline(y=0.5, line=dict(color=_PLACEHOLDER[ROLE_EVENT],
                                       width=1, dash="dot"))
    _layout(fig, height=190, ytitle="quality", xtitle="minutes into recording")
    fig.update_yaxes(range=[0, 1.02])
    return _fig_html(fig, include_js)


def _events_figure(summary, label, include_js):
    go = _plotly()
    ev = summary.events.get(label)
    fig = go.Figure()
    if ev is not None and ev.hourly_counts.size:
        centres = (ev.hour_edges[:-1] + ev.bin_s / 2) / 60.0
        fig.add_trace(go.Bar(
            x=centres, y=ev.hourly_counts,
            width=(ev.bin_s / 60.0) * 0.86,   # a real width; else one bin fills the axis
            marker_color=_PLACEHOLDER[ROLE_EVENT],
            hovertemplate="%{y} events<extra></extra>"))
    _layout(fig, height=220, ytitle="events", xtitle="minutes into recording")
    return _fig_html(fig, include_js)


def _waveforms_figure(examples, include_js):
    go = _plotly()
    fig = go.Figure()
    for centre, t, wave in examples:
        fig.add_trace(go.Scatter(
            x=t, y=wave, mode="lines",
            line=dict(width=1.2, color=_PLACEHOLDER[ROLE_EVENT]), opacity=.5,
            hoverinfo="skip"))
    fig.add_vline(x=0, line=dict(color=_PLACEHOLDER[ROLE_MUTED],
                                 width=1, dash="dot"))
    _layout(fig, height=240, ytitle="mV", xtitle="seconds from the flagged beat")
    return _fig_html(fig, include_js)


def _multichannel_figure(recording, peaks_by_sensor, window_s, include_js):
    go = _plotly()
    from plotly.subplots import make_subplots

    n = recording.n_channels
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=.07,
                        subplot_titles=[f"{c.sensor_id} — {c.position}"
                                        for c in recording])
    origin = recording.t_origin
    t0, t1 = window_s
    for r, ch in enumerate(recording, start=1):
        elapsed = ch.elapsed_since(origin)
        m = (elapsed >= t0) & (elapsed <= t1)
        fig.add_trace(go.Scattergl(
            x=elapsed[m], y=ch.signal[m], mode="lines",
            line=dict(color=_PLACEHOLDER[ROLE_TRACE], width=1.1),
            hoverinfo="skip"), row=r, col=1)
        pk = np.asarray(peaks_by_sensor.get(ch.sensor_id, []), dtype=int)
        pk = pk[(pk >= 0) & (pk < ch.n_samples)]
        if pk.size:
            pt = elapsed[pk]
            sel = (pt >= t0) & (pt <= t1)
            fig.add_trace(go.Scattergl(
                x=pt[sel], y=ch.signal[pk][sel], mode="markers",
                marker=dict(color=_PLACEHOLDER[ROLE_EVENT], size=7,
                            symbol="triangle-down"),
                hoverinfo="skip"), row=r, col=1)

    fig.update_layout(
        height=168 * n + 40, margin=dict(l=58, r=16, t=26, b=42),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_PLACEHOLDER[ROLE_MUTED], size=11),
        showlegend=False)
    fig.update_xaxes(gridcolor=_PLACEHOLDER[ROLE_GRID], zeroline=False,
                     linecolor=_PLACEHOLDER[ROLE_GRID])
    fig.update_yaxes(gridcolor=_PLACEHOLDER[ROLE_GRID], zeroline=False,
                     linecolor=_PLACEHOLDER[ROLE_GRID])
    fig.update_xaxes(title=dict(text="seconds from recording start",
                                font=dict(size=11)), row=n, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 11.5
        ann.font.color = _PLACEHOLDER[ROLE_MUTED]
    return _fig_html(fig, include_js)


# ------------------------------------------------------------------ page


CSS = """
:root{
  color-scheme:light;
  --bg:#f4f5f7; --card:#ffffff; --ink:#111820; --ink-2:#4a5563; --ink-3:#7a8694;
  --rule:#e1e5ea; --rule-2:#eef1f4;
  --accent:#b3283c; --trace:#2c7c9c; --good:#3f8a63; --warn:#9a6410;
  --grid:rgba(120,132,148,.20);
  --shadow:0 1px 2px rgba(17,24,32,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#0e1116; --card:#161a21; --ink:#e8ecf1; --ink-2:#a7b1bd; --ink-3:#727d8a;
    --rule:#262c35; --rule-2:#1c2129;
    --accent:#e8697c; --trace:#5fb3d4; --good:#68bd92; --warn:#d9a256;
    --grid:rgba(150,162,178,.18);
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0e1116; --card:#161a21; --ink:#e8ecf1; --ink-2:#a7b1bd; --ink-3:#727d8a;
  --rule:#262c35; --rule-2:#1c2129;
  --accent:#e8697c; --trace:#5fb3d4; --good:#68bd92; --warn:#d9a256;
  --grid:rgba(150,162,178,.18);
  --shadow:none;
}
*{box-sizing:border-box}
html{overflow-x:hidden}
body{
  margin:0; background:var(--bg); color:var(--ink); overflow-x:hidden;
  font:15px/1.6 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1060px; margin:0 auto; padding-inline:20px; padding-block:0 72px;
      min-width:0}

header{padding-block:32px 22px; border-bottom:1px solid var(--rule); margin-bottom:26px}
.kicker{
  font-size:11px; font-weight:650; letter-spacing:.1em; text-transform:uppercase;
  color:var(--accent); margin-bottom:7px;
}
h1{margin:0; font-size:clamp(21px,3.4vw,27px); font-weight:650; letter-spacing:-.02em}
.meta{margin:7px 0 0; color:var(--ink-3); font-size:13.5px}

h2{
  font-size:12px; font-weight:650; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink-3); margin:36px 0 12px;
}
h2:first-of-type{margin-top:0}

/* key figures: a compact ledger, not a wall of tiles */
.figures{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(118px,1fr));
  background:var(--card); border:1px solid var(--rule); border-radius:10px;
  overflow:hidden; box-shadow:var(--shadow);
}
.figures > div{padding:13px 15px; border-right:1px solid var(--rule-2)}
.figures > div:last-child{border-right:0}
.fk{font-size:10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase;
    color:var(--ink-3); margin-bottom:3px; white-space:nowrap}
.fv{font-size:20px; font-weight:650; letter-spacing:-.02em; font-variant-numeric:tabular-nums}
.fv.hl{color:var(--accent)}
.fn{font-size:11.5px; color:var(--ink-3); margin-top:2px; line-height:1.35}

.readme{
  margin-top:14px; background:var(--card); border:1px solid var(--rule);
  border-radius:10px; padding:15px 18px; box-shadow:var(--shadow);
}
.readme b{display:block; font-size:12.5px; margin-bottom:7px;
          letter-spacing:.02em; color:var(--warn)}
.readme ul{margin:0; padding-left:17px; color:var(--ink-2); font-size:13.5px}
.readme li{margin-bottom:4px}
.readme li:last-child{margin-bottom:0}

.panel{
  background:var(--card); border:1px solid var(--rule); border-radius:10px;
  padding:16px 18px 6px; margin-bottom:14px; box-shadow:var(--shadow);
  min-width:0;   /* see .two: without this a wide chart props the card open */
}
.panel h3{margin:0 0 2px; font-size:15px; font-weight:650; letter-spacing:-.01em}
.panel .cap{margin:0 0 6px; font-size:13px; color:var(--ink-2); max-width:78ch}
.plot{width:100%; min-width:0; overflow:hidden}

/* "1fr" is shorthand for minmax(auto, 1fr), and that auto minimum means a
   track will not shrink below the intrinsic width of what is inside it. A
   chart rendered wider than its column therefore props the column open
   instead of being constrained by it, the card overflows its card, and the
   page gains a horizontal scrollbar. minmax(0, 1fr) lets the track shrink,
   which is what allows the plot to be told how wide it may be.
   The same reasoning puts min-width:0 on the panel and the plot box. */
.two{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px}
@media (max-width:760px){.two{grid-template-columns:minmax(0,1fr)}}

.scroll{overflow-x:auto}
table{border-collapse:collapse; width:100%; font-size:13.5px; min-width:520px}
th{font-size:10.5px; font-weight:600; letter-spacing:.07em; text-transform:uppercase;
   color:var(--ink-3); text-align:left; padding:0 12px 8px 0;
   border-bottom:1px solid var(--rule)}
td{padding:10px 12px 10px 0; border-bottom:1px solid var(--rule-2)}
td.n{text-align:right; font-variant-numeric:tabular-nums; padding-right:0}
th.n{text-align:right; padding-right:0}
tr:last-child td{border-bottom:0}

.bar{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 0 12px}
.tabs{display:flex; gap:6px; flex-wrap:wrap}
.tab{
  font:inherit; font-size:12.5px; font-weight:550; padding:6px 13px;
  border:1px solid var(--rule); background:var(--card); color:var(--ink-2);
  border-radius:99px; cursor:pointer; transition:background .12s,color .12s;
}
.tab:hover{color:var(--ink)}
.tab.on{background:var(--ink); color:var(--bg); border-color:var(--ink)}
.tab:focus-visible,.ghost:focus-visible{outline:2px solid var(--trace); outline-offset:2px}
.ghost{
  font:inherit; font-size:12.5px; padding:6px 13px; margin-left:auto;
  border:1px solid var(--rule); background:var(--card); color:var(--ink-2);
  border-radius:99px; cursor:pointer;
}
.ghost:hover{color:var(--ink)}
.hint{font-size:12px; color:var(--ink-3)}
.panel.clickable{cursor:pointer}
.panel .tip{
  font-size:11.5px; color:var(--trace); font-weight:550;
  margin:0 0 6px; display:flex; align-items:center; gap:5px;
}
/* Motion is used to point at what changed, not to decorate. Panels fade as
   you move between sensors so a switch does not look like a redraw of the
   same data; the inspector slides a little when new waveforms load. Both are
   short enough to read as feedback rather than as an effect. */
@keyframes panel-in{from{opacity:0; transform:translateY(4px)} to{opacity:1; transform:none}}
[data-sensor-panel]:not([hidden]){animation:panel-in .22s ease-out}
.inspect-flash{animation:panel-in .26s ease-out}
.panel{transition:border-color .15s ease}
.panel.clickable:hover{border-color:var(--trace)}
@media (prefers-reduced-motion:reduce){
  *{transition:none!important; animation:none!important}
}
footer{margin-top:40px; padding-top:16px; border-top:1px solid var(--rule);
       color:var(--ink-3); font-size:12.5px}
.js-modebar{opacity:.5}
"""

def _payload(recording, summaries, event_label, bin_examples) -> str:
    """Data the page's script needs, as JSON.

    Only the waveforms behind each histogram bar and the bar counts: enough to
    answer "were those beats real", without shipping the whole recording.
    """
    counts = {}
    for sid, sm in summaries.items():
        ev = sm.events.get(event_label)
        counts[sid] = ev.hourly_counts.tolist() if ev is not None else []
    blob = json.dumps({
        "fs": float(recording[0].fs),
        "counts": counts,
        "examples": bin_examples or {},
    }, separators=(",", ":"))

    # JSON inside a <script> block is not safe by virtue of being JSON. A
    # sensor id containing "</script>" closes the block early and everything
    # after it is parsed as HTML -- script injection through a field that
    # merely names a piece of hardware. Escaping the three characters that can
    # start an HTML token closes that off; they are valid JSON escapes, so the
    # parsed value is unchanged.
    return (blob.replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace("&", "\\u0026"))


def _interactive_js() -> str:
    """The dashboard's behaviour, kept beside this module as real JavaScript.

    Held in a .js file rather than a Python string so it stays syntax-checked
    and diffable.
    """
    return (Path(__file__).with_name("interactive.js")).read_text(encoding="utf-8")


def _figure(key, value, note="", highlight=False):
    cls = "fv hl" if highlight else "fv"
    n = f'<div class="fn">{html.escape(note)}</div>' if note else ""
    return (f'<div><div class="fk">{html.escape(key)}</div>'
            f'<div class="{cls}">{html.escape(value)}</div>{n}</div>')


def _panel(title, caption, fig_html, link=None, events_for=None, tip=None):
    """One chart with its heading and explanation.

    ``link`` puts the plot in a shared time group so zooming one zooms all of
    them; ``events_for`` marks it as the clickable histogram for that sensor.
    The attributes land on the plot's own div, which is what the script binds
    Plotly event handlers to.
    """
    attrs = ""
    if link:
        attrs += f' data-link-time="{html.escape(link)}"'
    if events_for:
        attrs += f' data-events-for="{html.escape(events_for)}"'
    # The markers must sit on the graph div itself, not a wrapper.
    fig_html = fig_html.replace('class="plotly-graph-div"',
                                f'class="plotly-graph-div"{attrs}', 1)
    tip_html = (f'<p class="tip">&#9656; {html.escape(tip)}</p>') if tip else ""
    return (f'<div class="panel"><h3>{html.escape(title)}</h3>'
            f'<p class="cap">{caption}</p>{tip_html}'
            f'<div class="plot">{fig_html}</div></div>')


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
    bin_examples: dict | None = None,
) -> Path:
    """Render the dashboard to a self-contained HTML file.

    Args:
        recording: the ``SynchronizedRecording`` being reported on.
        summaries: ``{sensor_id: ChannelSummary}``.
        peaks_by_sensor: ``{sensor_id: peak indices}``.
        event_label: name of the abnormal-beat class summarised.
        examples: output of ``representative_beats`` for the primary sensor.
        caveats: what the numbers do not mean, rendered near the top rather
            than in a footnote. A dashboard showing an arrhythmia burden
            without saying how it was derived invites more trust than the
            evidence supports.
        inline_plotly: embed the plotting library (~3.5 MB) so the file opens
            offline. Tests turn it off; embedding it repeatedly makes the
            suite too slow to be a useful feedback loop.
        bin_examples: ``{sensor_id: [[{"t","y"}, ...], ...]}`` from
            ``examples_by_bin`` -- the waveforms behind each histogram bar,
            so a reader can click a spike and see whether those beats were
            real rather than taking the count on trust.
    """
    parts: list[str] = []
    first = inline_plotly

    primary_id = next(iter(summaries))
    primary = summaries[primary_id]
    ev = primary.events.get(event_label)
    total_beats = sum(s.n_beats for s in summaries.values())
    esc = html.escape

    parts.append('<div class="wrap"><header>')
    parts.append('<div class="kicker">Monitoring report</div>')
    parts.append(f"<h1>{esc(title)}</h1>")
    parts.append(f'<p class="meta">{esc(subtitle)}</p></header>')

    # ---------------------------------------------------------- figures
    parts.append("<h2>At a glance</h2>")
    parts.append('<div class="figures">')
    parts.append(_figure("Duration", f"{primary.duration_s / 3600:.2f} h"))
    parts.append(_figure(
        "Analysed", f"{primary.coverage.fraction * 100:.0f}%",
        f"{primary.coverage.n_gaps} gap(s)"))
    parts.append(_figure("Beats", f"{total_beats:,}"))
    parts.append(_figure("Mean HR", f"{primary.mean_hr:.0f} bpm",
                         f"{primary.min_hr:.0f}–{primary.max_hr:.0f}"))
    if ev is not None:
        note = f"{ev.per_hour:.0f}/h"
        if ev.rate_is_extrapolated:
            note = f"{ev.per_hour:.0f}/h projected"
        parts.append(_figure(f"{event_label}s", f"{ev.count:,}", highlight=True))
        parts.append(_figure("Burden", f"{ev.burden_pct:.1f}%", note, highlight=True))
    parts.append(_figure("SDNN", f"{primary.sdnn_ms:.0f} ms"))
    parts.append("</div>")

    if caveats:
        parts.append('<div class="readme"><b>Before reading the numbers</b><ul>')
        for c in caveats:
            parts.append(f"<li>{esc(c)}</li>")
        parts.append("</ul></div>")

    # ----------------------------------------------------------- trends
    sensor_ids = list(summaries)
    parts.append("<h2>Trends and events</h2>")
    parts.append('<div class="bar">')
    if len(sensor_ids) > 1:
        parts.append('<div class="tabs" role="tablist">')
        for i, sid in enumerate(sensor_ids):
            on = " on" if i == 0 else ""
            parts.append(
                f'<button class="tab{on}" role="tab" data-sensor-tab="{esc(sid)}" '
                f'aria-selected="{str(i == 0).lower()}">{esc(sid)}</button>')
        parts.append("</div>")
    parts.append('<span class="hint" id="zoom-note-trends"></span>')
    parts.append('<button class="ghost" id="theme-btn" type="button">Dark</button>')
    parts.append("</div>")

    for i, sid in enumerate(sensor_ids):
        sm = summaries[sid]
        sev = sm.events.get(event_label)
        hidden = "" if i == 0 else " hidden"
        parts.append(f'<div data-sensor-panel="{esc(sid)}"{hidden}>')

        parts.append(_panel(
            "Heart rate",
            "Average beats per minute in each one-minute window. Breaks in the "
            "line are stretches with no usable signal - left empty rather than "
            "bridged, so a gap cannot be read as a steady rate. Drag across any "
            "chart to zoom; the others follow.",
            _hr_figure(sm, first), link="trends"))
        first = False

        parts.append('<div class="two">')
        parts.append(_panel(
            "RR intervals",
            "One dot per heartbeat: the time since the previous beat. A tight "
            "band is a regular rhythm; scatter means the spacing is varying. "
            "Ectopic beats show as a low dot followed by a high one - an early "
            "beat, then a compensatory pause.",
            _rr_figure(sm, False), link="trends"))
        parts.append(_panel(
            "Signal quality",
            "How trustworthy the signal is, scored every 10 seconds. Below the "
            "dotted line at 0.5 it is treated as unusable and excluded from the "
            "counts above.",
            _sqi_figure(sm, False), link="trends"))
        parts.append("</div>")

        if sev is not None:
            unit = "hour" if sev.bin_s >= 3600 else f"{int(sev.bin_s / 60)}-minute"
            parts.append(_panel(
                f"When {event_label}s happened",
                f"Flagged beats in each {unit} block. This is what shows whether "
                f"events spread evenly through the recording or cluster into "
                f"episodes, which is usually the more interesting pattern.",
                _events_figure(sm, event_label, False),
                events_for=sid,
                tip="Click a bar to see the beats behind it"))

            parts.append(
                f'<div class="panel"><h3>Beats behind the bar</h3>'
                f'<p class="cap" id="inspect-cap-{esc(sid)}">'
                f'Click a bar above to load the waveforms from that block.</p>'
                f'<div class="plot" id="inspect-{esc(sid)}"></div></div>')

        parts.append("</div>")

    # ---------------------------------------------------------- sensors
    parts.append("<h2>Per-sensor comparison</h2>")
    parts.append('<div class="panel"><h3>Summary by sensor</h3>')
    parts.append('<p class="cap">The same heart seen from each recording '
                 'position. Differences in beat count or burden between '
                 'sensors are a signal in themselves: they mean the positions '
                 'are not seeing the same events equally well.</p>')
    parts.append('<div class="scroll"><table><tr><th>Sensor</th><th>Position</th>'
                 '<th class="n">Coverage</th><th class="n">Beats</th>'
                 f'<th class="n">Mean HR</th><th class="n">{esc(event_label)}</th>'
                 '<th class="n">Burden</th></tr>')
    for sid, s in summaries.items():
        e = s.events.get(event_label)
        parts.append(
            f"<tr><td>{esc(sid)}</td><td>{esc(s.position)}</td>"
            f'<td class="n">{s.coverage.fraction * 100:.0f}%</td>'
            f'<td class="n">{s.n_beats:,}</td>'
            f'<td class="n">{s.mean_hr:.0f}</td>'
            f'<td class="n">{e.count if e else 0:,}</td>'
            f'<td class="n">{e.burden_pct if e else 0:.1f}%</td></tr>')
    parts.append("</table></div></div>")

    t0, t1 = trace_window_s
    parts.append(_panel(
        "Synchronised traces",
        f"The raw signal from every sensor between {t0:g} and {t1:g} seconds, on "
        f"one shared clock. Triangles mark detected beats. Because the axis is "
        f"time from the recording start rather than each sensor's own first "
        f"sample, sensors that started at different moments appear correctly "
        f"offset instead of falsely aligned.",
        _multichannel_figure(recording, peaks_by_sensor, trace_window_s, False)))

    parts.append(
        "<footer>Generated by ecgmon. Research prototype — not a medical device "
        "and not for diagnostic use.</footer></div>")

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>"
        + "".join(parts)
        + f"<script>window.__ECG__={_payload(recording, summaries, event_label, bin_examples)};</script>"
        + f"<script>{_interactive_js()}</script></body></html>"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
