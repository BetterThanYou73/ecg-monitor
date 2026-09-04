"""Portable long-term ECG monitoring: signal analysis pipeline.

Designed around independent single-channel sensors recorded synchronously at
different anatomical positions. Single-sensor operation is the one-channel
case of the same model, so the multi-sensor path needs no rewrite later.
"""

from .io.record import (
    ChannelRecord,
    SynchronizedRecording,
    UPPER_CHEST,
    LOWER_CHEST,
    LEFT_LATERAL,
    RIGHT_LATERAL,
)

__version__ = "0.1.0"

__all__ = [
    "ChannelRecord",
    "SynchronizedRecording",
    "UPPER_CHEST",
    "LOWER_CHEST",
    "LEFT_LATERAL",
    "RIGHT_LATERAL",
]
