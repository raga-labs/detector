import numpy as np

from raga_detector.features import TDMS_BINS, feature_windows, tdms


def test_tdms_is_normalized() -> None:
    times = np.arange(0, 10, 0.01)
    frequencies = 220 * 2 ** (100 * np.sin(times * 4) / 1200)
    surface = tdms(times, frequencies, tonic_hz=220)
    assert surface.shape == (TDMS_BINS * TDMS_BINS,)
    assert np.isclose(surface.sum(), 1)


def test_short_clip_is_rejected() -> None:
    times = np.arange(0, 4, 0.01)
    frequencies = np.full(times.shape, 220)
    assert feature_windows(times, frequencies, tonic_hz=220) == []
