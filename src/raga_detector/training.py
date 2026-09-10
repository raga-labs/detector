from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .compmusic import PitchRecording, load_pitch
from .features import feature_windows


def split_by_artist(
    recordings: list[PitchRecording], test_fraction: float = 0.25, seed: int = 42
) -> tuple[list[PitchRecording], list[PitchRecording]]:
    """Choose a deterministic split with no artist appearing on both sides."""
    artists = sorted({recording.artist for recording in recordings})
    ragas = sorted({recording.raga for recording in recordings})
    desired = Counter(
        {
            raga: max(1, round(sum(r.raga == raga for r in recordings) * test_fraction))
            for raga in ragas
        }
    )
    test_artist_count = max(1, round(len(artists) * test_fraction))
    best_artists: set[str] | None = None
    best_score: tuple[int, int, int] | None = None

    rng = random.Random(seed)
    for _ in range(5_000):
        candidate = set(rng.sample(artists, test_artist_count))
        test_counts = Counter(r.raga for r in recordings if r.artist in candidate)
        train_counts = Counter(r.raga for r in recordings if r.artist not in candidate)
        valid_classes = sum(test_counts[raga] > 0 and train_counts[raga] > 0 for raga in ragas)
        minimum_test = min(test_counts[raga] for raga in ragas)
        distance = sum(abs(test_counts[raga] - desired[raga]) for raga in ragas)
        score = (valid_classes, -distance, minimum_test)
        if best_score is None or score > best_score:
            best_score = score
            best_artists = candidate

    if best_artists is None or best_score is None or best_score[0] != len(ragas):
        raise ValueError("Could not create an artist-disjoint split covering every raga")

    train = [recording for recording in recordings if recording.artist not in best_artists]
    test = [recording for recording in recordings if recording.artist in best_artists]
    return train, test


def _limited_windows(recording: PitchRecording, maximum: int) -> list[np.ndarray]:
    times, frequencies, tonic_hz = load_pitch(recording)
    windows = feature_windows(times, frequencies, tonic_hz)
    if len(windows) <= maximum:
        return windows
    indices = np.linspace(0, len(windows) - 1, maximum, dtype=int)
    return [windows[index] for index in indices]


def train_model(
    train_recordings: list[PitchRecording],
    test_recordings: list[PitchRecording],
    output_dir: Path,
    max_windows_per_recording: int = 40,
    seed: int = 42,
) -> dict:
    from xgboost import XGBClassifier

    classes = sorted({recording.raga for recording in train_recordings})
    class_index = {raga: index for index, raga in enumerate(classes)}
    train_features: list[np.ndarray] = []
    train_labels: list[int] = []

    for number, recording in enumerate(train_recordings, start=1):
        windows = _limited_windows(recording, max_windows_per_recording)
        train_features.extend(windows)
        train_labels.extend([class_index[recording.raga]] * len(windows))
        print(
            f"features {number:3}/{len(train_recordings)} "
            f"{recording.raga:20} {recording.artist:28} windows={len(windows):2}",
            flush=True,
        )

    if not train_features:
        raise ValueError("No usable training windows were created")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(np.vstack(train_features), np.asarray(train_labels))

    correct_top1 = 0
    correct_top3 = 0
    evaluated: list[dict] = []
    for recording in test_recordings:
        windows = _limited_windows(recording, max_windows_per_recording)
        if not windows:
            continue
        probabilities = model.predict_proba(np.vstack(windows)).mean(axis=0)
        order = np.argsort(probabilities)[::-1]
        prediction = classes[int(order[0])]
        top3 = [classes[int(index)] for index in order[:3]]
        correct_top1 += prediction == recording.raga
        correct_top3 += recording.raga in top3
        evaluated.append(
            {
                "track_id": recording.track_id,
                "artist": recording.artist,
                "expected": recording.raga,
                "predicted": prediction,
                "top3": top3,
            }
        )

    total = len(evaluated)
    metrics = {
        "classes": classes,
        "train_recordings": len(train_recordings),
        "test_recordings": total,
        "train_artists": len({recording.artist for recording in train_recordings}),
        "test_artists": len({recording.artist for recording in test_recordings}),
        "training_windows": len(train_features),
        "top1_accuracy": correct_top1 / total if total else 0.0,
        "top3_accuracy": correct_top3 / total if total else 0.0,
        "evaluated": evaluated,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "raaga_xgb.json")
    (output_dir / "raaga_xgb.classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "raaga_xgb.calib.json").write_text('{"temperature": 1.0}\n')
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    return metrics
