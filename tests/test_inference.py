import numpy as np
import soundfile as sf

from raga_detector.inference import SAMPLE_RATE, extract_features


def test_short_audio_is_rejected_before_heavy_extractors(tmp_path) -> None:
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    windows, tonic, seconds = extract_features(path)
    assert windows == []
    assert tonic == 0
    assert seconds == 1
