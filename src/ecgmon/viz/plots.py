"""Diagnostic plots.

Two figures matter at this stage. ``plot_detection_overview`` shows one
channel with its detections and the Pan-Tompkins internals, which is how you
tell a bad electrode from a bad threshold. ``plot_multichannel`` puts every
sensor on one shared time axis: the same cardiac event seen from several
positions, lined up.

Matplotlib is used here for static figures. Interactive Plotly versions
belong with the dashboard work, not with pipeline diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _lazy_mpl():
    import matplotlib

    matplotlib.use("Agg")  # no display needed; we always write to file
    import matplotlib.pyplot as plt

    return plt


def plot_detection_overview(
    ch,
    peaks: np.ndarray,
    stages: dict | None = None,
    window_s: tuple[float, float] = (0.0, 10.0),
    out_path: str | Path = "outputs/detection_overview.png",
    title: str | None = None,
) -> Path:
    """Plot a window of one channel with R-peak detections and PT internals.

    Args:
        ch: ``ChannelRecord`` to plot.
        peaks: R-peak sample indices for that channel.
        stages: Optional dict from ``pan_tompkins_stages`` for the lower panels.
        window_s: Time window to display, in seconds from the channel start.
        out_path: Where to write the PNG.
    """
    plt = _lazy_mpl()

    t0, t1 = window_s
    i0 = max(0, int(round(t0 * ch.fs)))
    i1 = min(ch.n_samples, int(round(t1 * ch.fs)))
    t = np.arange(i0, i1) / ch.fs

    in_window = peaks[(peaks >= i0) & (peaks < i1)]

    n_panels = 1 if stages is None else 3
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 3.2 * n_panels), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(t, ch.signal[i0:i1], linewidth=0.9, color="#1f2a44", label="ECG")
    if in_window.size:
        ax.plot(
            in_window / ch.fs,
            ch.signal[in_window],
            "v",
            markersize=7,
            color="#d1495b",
            label=f"R peaks (n={in_window.size})",
        )
    ax.set_ylabel(f"amplitude ({ch.units})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title(
        title
        or f"{ch.sensor_id}  |  {ch.position}  |  {ch.fs:g} Hz  |  {t0:g}-{t1:g}s"
    )

    if stages is not None:
        axes[1].plot(t, stages["bandpassed"][i0:i1], linewidth=0.9, color="#2a6f97")
        axes[1].set_ylabel("bandpassed\n5-15 Hz")
        axes[1].grid(alpha=0.25)

        axes[2].plot(t, stages["integrated"][i0:i1], linewidth=0.9, color="#3a7d44")
        axes[2].set_ylabel("integrated\n(150 ms)")
        axes[2].grid(alpha=0.25)

    axes[-1].set_xlabel("time (s)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_multichannel(
    recording,
    peaks_by_sensor: dict | None = None,
    window_s: tuple[float, float] = (0.0, 10.0),
    out_path: str | Path = "outputs/multichannel.png",
) -> Path:
    """Stack every channel on one shared reference timeline.

    The x axis is seconds from ``recording.t_origin``, not from each channel's
    own first sample, so channels that started at different moments appear
    correctly offset rather than falsely aligned. This is the property that
    has to hold before any cross-sensor claim can be made.
    """
    plt = _lazy_mpl()

    t0, t1 = window_s
    origin = recording.t_origin
    n = recording.n_channels

    fig, axes = plt.subplots(
        n, 1, figsize=(14, 2.6 * n), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, ch in zip(axes, recording):
        elapsed = ch.elapsed_since(origin)
        mask = (elapsed >= t0) & (elapsed <= t1)
        ax.plot(elapsed[mask], ch.signal[mask], linewidth=0.9, color="#1f2a44")

        if peaks_by_sensor and ch.sensor_id in peaks_by_sensor:
            pk = np.asarray(peaks_by_sensor[ch.sensor_id], dtype=int)
            pk = pk[(pk >= 0) & (pk < ch.n_samples)]
            pk_t = elapsed[pk]
            sel = (pk_t >= t0) & (pk_t <= t1)
            ax.plot(pk_t[sel], ch.signal[pk][sel], "v", markersize=6, color="#d1495b")

        ax.set_ylabel(f"{ch.sensor_id}\n{ch.position}", fontsize=8)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(f"seconds from recording origin ({origin.isoformat()})")
    axes[0].set_title(
        f"{recording.record_id}: {n} synchronized channel(s), {t0:g}-{t1:g}s"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_hr_trend(
    centres: np.ndarray,
    hr: np.ndarray,
    out_path: str | Path = "outputs/hr_trend.png",
    title: str = "Heart-rate trend",
) -> Path:
    """Plot a heart-rate trend line, leaving gaps where data is missing."""
    plt = _lazy_mpl()

    fig, ax = plt.subplots(figsize=(14, 3.5), constrained_layout=True)
    ax.plot(centres / 60.0, hr, linewidth=1.2, color="#2a6f97")
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("heart rate (bpm)")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
