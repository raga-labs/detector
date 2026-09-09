from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import feature_windows

SAMPLE_RATE = 44_100
PITCH_HOP = 128


def extract_features(audio_path: Path) -> tuple[list[np.ndarray], float, float]:
    import librosa

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=90)
    if len(audio) < SAMPLE_RATE * 5:
        return [], 0.0, len(audio) / SAMPLE_RATE

    from compiam.melody.tonic_identification.tonic_multipitch import TonicIndianMultiPitch
    from essentia.standard import PredominantPitchMelodia

    tonic = float(TonicIndianMultiPitch().extract(audio, input_sr=SAMPLE_RATE))
    pitch, _ = PredominantPitchMelodia(
        hopSize=PITCH_HOP,
        sampleRate=SAMPLE_RATE,
    )(audio)
    times = np.arange(len(pitch)) * PITCH_HOP / SAMPLE_RATE
    return feature_windows(times, pitch, tonic), tonic, len(audio) / SAMPLE_RATE


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max()
    scaled = np.exp(logits)
    return scaled / scaled.sum()


def predict(audio_path: Path, model_dir: Path) -> dict:
    from xgboost import XGBClassifier

    windows, tonic, seconds = extract_features(audio_path)
    if not windows:
        raise ValueError("Could not extract at least five seconds of clear melody")

    classes = json.loads((model_dir / "raaga_xgb.classes.json").read_text())
    calibration = json.loads((model_dir / "raaga_xgb.calib.json").read_text())
    model = XGBClassifier()
    model.load_model(model_dir / "raaga_xgb.json")

    probabilities = model.predict_proba(np.vstack(windows)).mean(axis=0)
    probabilities = _temperature_scale(probabilities, calibration["temperature"])
    order = np.argsort(probabilities)[::-1][:3]
    return {
        "tonic_hz": round(tonic, 2),
        "analysed_seconds": round(seconds, 2),
        "windows": len(windows),
        "predictions": [
            {"raga": classes[i], "confidence": round(float(probabilities[i]), 4)}
            for i in order
        ],
    }
