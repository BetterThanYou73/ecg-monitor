"""Ingestion of PhysioNet/WFDB records into the project data model.

MIT-BIH records carry two simultaneously-recorded leads. That is not the same
thing as two of our sensors -- the leads share an amplifier and a clock,
whereas our sensors will not -- but mapping each lead onto its own
``ChannelRecord`` means the multi-channel code path is exercised against real
annotated data now, rather than first being tested when hardware arrives.

Where the analogy breaks down is exactly where the interesting work is:
``clock_offset_s`` is always 0.0 here, and will not be for real sensors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .record import ChannelRecord, SynchronizedRecording, UNKNOWN_POSITION

# Beat annotation symbols used by MIT-BIH, grouped by what they mean for us.
# Reference: PhysioNet WFDB annotation codes.
NORMAL_BEATS = set("NLRej")
SUPRAVENTRICULAR_BEATS = set("AaJS")
VENTRICULAR_BEATS = set("VE")
FUSION_BEATS = set("F")
UNKNOWN_BEATS = set("/fQ")
BEAT_SYMBOLS = (
    NORMAL_BEATS | SUPRAVENTRICULAR_BEATS | VENTRICULAR_BEATS
    | FUSION_BEATS | UNKNOWN_BEATS
)

DEFAULT_DB = "mitdb"


@dataclass
class BeatAnnotations:
    """Reference beat labels accompanying a WFDB record."""

    sample: np.ndarray   # sample index of each annotation
    symbol: np.ndarray   # annotation character
    fs: float

    def beats_only(self) -> "BeatAnnotations":
        """Keep beat annotations, dropping rhythm and quality markers.

        Non-beat annotations mark things like rhythm changes and signal
        quality. Leaving them in would inflate the reference beat count and
        make a detector look worse than it is.
        """
        mask = np.isin(self.symbol, list(BEAT_SYMBOLS))
        return BeatAnnotations(self.sample[mask], self.symbol[mask], self.fs)

    def of_class(self, symbols: set) -> np.ndarray:
        """Sample indices of beats whose symbol is in ``symbols``."""
        return self.sample[np.isin(self.symbol, list(symbols))]

    @property
    def ventricular(self) -> np.ndarray:
        """PVC and ventricular escape beats -- the first detection target."""
        return self.of_class(VENTRICULAR_BEATS)

    def counts(self) -> dict:
        uniq, n = np.unique(self.symbol, return_counts=True)
        return dict(zip(uniq.tolist(), n.tolist()))

    def __len__(self) -> int:
        return int(self.sample.size)


def _configure_tls() -> None:
    """Point requests at the project CA bundle if one has been generated.

    Local TLS interception (corporate proxies, some antivirus products)
    breaks PhysioNet downloads with a certificate error unless the
    intercepting root is trusted.
    """
    bundle = Path(__file__).resolve().parents[3] / "configs" / "ca-bundle.pem"
    if bundle.exists():
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(bundle))
        os.environ.setdefault("SSL_CERT_FILE", str(bundle))


def _position_for_lead(lead_name: str) -> str:
    """Best-effort anatomical position for a WFDB lead name.

    Deliberately coarse. These are electrode configurations, not our sensor
    placements, and pretending otherwise would bake a wrong assumption into
    the data model.
    """
    name = lead_name.strip().upper()
    if name in {"MLII", "II", "ML2"}:
        return "lead_MLII"
    if name.startswith("V"):
        return f"lead_{name}"
    return f"lead_{name.lower()}" if name else UNKNOWN_POSITION


def load_wfdb_record(
    record_name: str,
    db: str | None = DEFAULT_DB,
    data_dir: str | Path | None = None,
    channels: list[int] | None = None,
) -> SynchronizedRecording:
    """Load one WFDB record as a ``SynchronizedRecording``.

    Args:
        record_name: Record id, e.g. ``"100"`` for MIT-BIH.
        db: PhysioNet database slug to download from, e.g. ``"mitdb"``.
            Pass ``None`` to read purely from ``data_dir``.
        data_dir: Local directory holding (or to cache) the record files.
        channels: Signal indices to load; all channels when omitted.

    Returns:
        A recording with one ``ChannelRecord`` per WFDB signal.
    """
    import wfdb

    _configure_tls()

    kwargs = {}
    if channels is not None:
        kwargs["channels"] = channels

    if data_dir is not None:
        local = Path(data_dir) / record_name
        if local.with_suffix(".hea").exists():
            rec = wfdb.rdrecord(str(local), **kwargs)
        else:
            rec = wfdb.rdrecord(record_name, pn_dir=db, **kwargs)
    else:
        rec = wfdb.rdrecord(record_name, pn_dir=db, **kwargs)

    signals = np.asarray(rec.p_signal, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, None]

    names = list(rec.sig_name or [])
    units = list(rec.units or [])
    fs = float(rec.fs)

    # WFDB records rarely carry a wall-clock start; anchor at epoch so that
    # timestamps stay well-defined and comparable.
    t0 = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if getattr(rec, "base_date", None) and getattr(rec, "base_time", None):
        t0 = datetime.combine(rec.base_date, rec.base_time, tzinfo=timezone.utc)

    chans = []
    for i in range(signals.shape[1]):
        lead = names[i] if i < len(names) else f"ch{i}"
        col = signals[:, i]
        # Records occasionally contain NaN where a lead was disconnected.
        if np.isnan(col).any():
            col = np.nan_to_num(col, nan=float(np.nanmedian(col)))
        chans.append(
            ChannelRecord(
                sensor_id=f"{record_name}_{lead}",
                fs=fs,
                signal=col,
                position=_position_for_lead(lead),
                t_start=t0,
                units=units[i] if i < len(units) else "mV",
                label=lead,
            )
        )

    return SynchronizedRecording(
        record_id=str(record_name),
        channels=chans,
        subject_id=str(record_name),
        source=f"physionet/{db}" if db else str(data_dir),
        metadata={"n_sig": signals.shape[1], "comments": list(rec.comments or [])},
    )


def load_annotations(
    record_name: str,
    db: str | None = DEFAULT_DB,
    data_dir: str | Path | None = None,
    extension: str = "atr",
) -> BeatAnnotations:
    """Load reference annotations for a record."""
    import wfdb

    _configure_tls()

    if data_dir is not None:
        local = Path(data_dir) / record_name
        if local.with_suffix(f".{extension}").exists():
            ann = wfdb.rdann(str(local), extension)
        else:
            ann = wfdb.rdann(record_name, extension, pn_dir=db)
    else:
        ann = wfdb.rdann(record_name, extension, pn_dir=db)

    return BeatAnnotations(
        sample=np.asarray(ann.sample, dtype=np.int64),
        symbol=np.asarray(ann.symbol),
        fs=float(ann.fs),
    )


def download_records(
    record_names: list[str],
    db: str = DEFAULT_DB,
    data_dir: str | Path = "data/raw",
) -> Path:
    """Cache records locally so later runs work offline."""
    import wfdb

    _configure_tls()
    target = Path(data_dir) / db
    target.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(db, str(target), records=record_names)
    return target
