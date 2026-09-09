from __future__ import annotations

import numpy as np

TDMS_BINS = 48
TDMS_DELAY_SECONDS = 0.3
WINDOW_SECONDS = 30.0
HOP_SECONDS = 15.0
MIN_VOICED_FRACTION = 0.5


def tdms(times, frequencies, tonic_hz: float) -> np.ndarray:
    """Return a tonic-normalized time-delayed melody surface."""
    times = np.asarray(times, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    empty = np.zeros(TDMS_BINS * TDMS_BINS, dtype=np.float32)
    if times.size < 2 or tonic_hz <= 0:
        return empty

    voiced = frequencies > 0
    cents = np.zeros_like(frequencies)
    cents[voiced] = np.mod(1200 * np.log2(frequencies[voiced] / tonic_hz), 1200)
    bins = np.mod((cents / (1200 / TDMS_BINS)).astype(int), TDMS_BINS)

    hop = float(np.median(np.diff(times)))
    delay_frames = max(1, int(round(TDMS_DELAY_SECONDS / hop)))
    if delay_frames >= len(bins):
        return empty

    valid = voiced[:-delay_frames] & voiced[delay_frames:]
    before = bins[:-delay_frames][valid]
    after = bins[delay_frames:][valid]
    if before.size == 0:
        return empty

    surface = np.zeros((TDMS_BINS, TDMS_BINS), dtype=np.float64)
    np.add.at(surface, (before, after), 1)
    return (surface / surface.sum()).astype(np.float32).ravel()


def feature_windows(times, frequencies, tonic_hz: float) -> list[np.ndarray]:
    times = np.asarray(times, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if times.size == 0 or times[-1] < 5:
        return []

    windows: list[np.ndarray] = []
    start = 0.0
    while start < times[-1]:
        mask = (times >= start) & (times < start + WINDOW_SECONDS)
        if mask.sum() == 0 or times[mask][-1] - times[mask][0] < 5:
            break
        if float((frequencies[mask] > 0).mean()) >= MIN_VOICED_FRACTION:
            surface = tdms(times[mask], frequencies[mask], tonic_hz)
            if surface.sum() > 0:
                windows.append(surface)
        start += HOP_SECONDS
    return windows
