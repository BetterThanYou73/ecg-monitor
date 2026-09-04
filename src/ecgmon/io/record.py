"""Core ECG data model.

The hardware this project targets is a set of *independent single-channel*
sensors that can be placed at different anatomical positions and recorded
synchronously. A single-sensor recording is therefore modelled as the N=1
case of a synchronized multi-sensor recording, so that nothing downstream
has to be rewritten when additional sensors are introduced.

Every channel carries its own sampling rate, its own absolute start time and
its own clock-offset correction, because separate sensors do not share a
clock and will drift relative to one another over a long recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

import numpy as np


# Anatomical placements we expect to use. Free-form strings are still allowed;
# these exist so that the common cases are spelled consistently.
UPPER_CHEST = "upper_chest"
LOWER_CHEST = "lower_chest"
LEFT_LATERAL = "left_lateral_chest"
RIGHT_LATERAL = "right_lateral_chest"
UNKNOWN_POSITION = "unknown"


@dataclass
class ChannelRecord:
    """One continuous single-channel ECG signal from one sensor.

    Attributes:
        sensor_id: Stable identifier of the physical sensor.
        fs: Sampling rate in Hz.
        signal: 1-D float array of samples.
        position: Anatomical placement of the sensor for this recording.
        t_start: Absolute timestamp of ``signal[0]``.
        units: Physical units of ``signal``.
        clock_offset_s: Correction added to this channel's timestamps to bring
            it onto the recording's reference clock. Estimated by
            synchronisation, not reported by the sensor.
        label: Human-readable channel name (e.g. the MIT-BIH lead name).
    """

    sensor_id: str
    fs: float
    signal: np.ndarray
    position: str = UNKNOWN_POSITION
    t_start: datetime = field(
        default_factory=lambda: datetime(1970, 1, 1, tzinfo=timezone.utc)
    )
    units: str = "mV"
    clock_offset_s: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=np.float64).ravel()
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        if self.t_start.tzinfo is None:
            self.t_start = self.t_start.replace(tzinfo=timezone.utc)

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def t_start_corrected(self) -> datetime:
        """Start time on the recording's reference clock."""
        return self.t_start + timedelta(seconds=self.clock_offset_s)

    @property
    def t_end_corrected(self) -> datetime:
        return self.t_start_corrected + timedelta(seconds=self.duration_s)

    def rel_time(self) -> np.ndarray:
        """Seconds from this channel's own start."""
        return np.arange(self.n_samples, dtype=np.float64) / self.fs

    def elapsed_since(self, origin: datetime) -> np.ndarray:
        """Seconds of each sample relative to ``origin`` on the reference clock.

        This is the axis to plot against when comparing several channels: it
        folds each sensor's start time and clock offset into one shared
        timeline.
        """
        lead_in = (self.t_start_corrected - origin).total_seconds()
        return lead_in + self.rel_time()

    def slice_seconds(self, start_s: float, end_s: float) -> "ChannelRecord":
        """Return a copy covering ``[start_s, end_s)`` from this channel's start."""
        i0 = max(0, int(round(start_s * self.fs)))
        i1 = min(self.n_samples, int(round(end_s * self.fs)))
        if i1 <= i0:
            raise ValueError(f"empty slice [{start_s}, {end_s}) for {self.sensor_id}")
        return replace(
            self,
            signal=self.signal[i0:i1].copy(),
            t_start=self.t_start + timedelta(seconds=i0 / self.fs),
        )

    def with_signal(self, signal: np.ndarray) -> "ChannelRecord":
        """Copy carrying a new signal of the same length (for filter stages)."""
        signal = np.asarray(signal, dtype=np.float64).ravel()
        if signal.size != self.n_samples:
            raise ValueError(
                f"length changed: {self.n_samples} -> {signal.size}. "
                "Use slice_seconds() for trimming."
            )
        return replace(self, signal=signal)

    def __repr__(self) -> str:
        return (
            f"ChannelRecord(sensor_id={self.sensor_id!r}, position={self.position!r}, "
            f"fs={self.fs:g}Hz, n={self.n_samples}, dur={self.duration_s:.1f}s)"
        )


@dataclass
class SynchronizedRecording:
    """One or more channels recorded simultaneously from the same subject.

    A conventional single-sensor recording is the ``len(channels) == 1`` case.
    Code written against this class works unchanged when more sensors are added.
    """

    record_id: str
    channels: list[ChannelRecord]
    subject_id: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("a recording needs at least one channel")
        seen: set[str] = set()
        for ch in self.channels:
            if ch.sensor_id in seen:
                raise ValueError(f"duplicate sensor_id {ch.sensor_id!r}")
            seen.add(ch.sensor_id)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def is_single_channel(self) -> bool:
        return self.n_channels == 1

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self):
        return iter(self.channels)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.channels[key]
        for ch in self.channels:
            if ch.sensor_id == key:
                return ch
        raise KeyError(f"no channel with sensor_id {key!r}")

    @property
    def t_origin(self) -> datetime:
        """Reference clock zero: the earliest corrected channel start."""
        return min(ch.t_start_corrected for ch in self.channels)

    @property
    def t_end(self) -> datetime:
        return max(ch.t_end_corrected for ch in self.channels)

    @property
    def duration_s(self) -> float:
        return (self.t_end - self.t_origin).total_seconds()

    def overlap_window(self) -> tuple[float, float]:
        """Span, in seconds from ``t_origin``, covered by *every* channel.

        Returns ``(0.0, 0.0)`` when the channels never overlap, which is the
        case to check for before any cross-sensor comparison.
        """
        origin = self.t_origin
        start = max((ch.t_start_corrected - origin).total_seconds() for ch in self)
        end = min((ch.t_end_corrected - origin).total_seconds() for ch in self)
        return (start, end) if end > start else (0.0, 0.0)

    @property
    def sampling_rates(self) -> set:
        return {ch.fs for ch in self.channels}

    @property
    def has_uniform_fs(self) -> bool:
        return len(self.sampling_rates) == 1

    def positions(self) -> dict:
        return {ch.sensor_id: ch.position for ch in self.channels}

    def map_channels(self, fn) -> "SynchronizedRecording":
        """Apply ``fn(ChannelRecord) -> ChannelRecord`` to every channel."""
        return replace(self, channels=[fn(ch) for ch in self.channels])

    def summary(self) -> str:
        lines = [
            f"{self.record_id} ({self.source or 'unknown source'}) - "
            f"{self.n_channels} channel(s), {self.duration_s:.1f}s"
        ]
        for ch in self.channels:
            lines.append(
                f"  {ch.sensor_id:<12} {ch.position:<20} "
                f"{ch.fs:g} Hz  {ch.duration_s:8.1f}s  {ch.label}"
            )
        if self.n_channels > 1:
            start, end = self.overlap_window()
            lines.append(f"  overlap: {start:.3f}s -> {end:.3f}s")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SynchronizedRecording({self.record_id!r}, "
            f"n_channels={self.n_channels}, duration={self.duration_s:.1f}s)"
        )
