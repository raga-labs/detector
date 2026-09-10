from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .compmusic import PitchRecording
from .training import (
    build_feature_cache,
    evaluate_classifier,
    fit_classifier,
    save_model_bundle,
)


def cross_validation_splits(
    recordings: list[PitchRecording], folds: int = 5, seed: int = 42
) -> list[tuple[list[PitchRecording], list[PitchRecording]]]:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = [recording.raga for recording in recordings]
    groups = [recording.artist for recording in recordings]
    indices = np.arange(len(recordings))
    result = []
    for train_indices, test_indices in splitter.split(indices, labels, groups):
        train = [recordings[int(index)] for index in train_indices]
        test = [recordings[int(index)] for index in test_indices]
        result.append((train, test))
    return result


def _normalized_title(recording: PitchRecording) -> str:
    title = Path(recording.track_id).name
    decomposed = unicodedata.normalize("NFKD", title)
    return "".join(character.lower() for character in decomposed if character.isalnum())


def cross_validate(
    recordings: list[PitchRecording],
    output_dir: Path,
    folds: int = 5,
    max_windows_per_recording: int = 40,
    seed: int = 42,
) -> dict:
    classes = sorted({recording.raga for recording in recordings})
    print("Loading TDMS features once for all folds...", flush=True)
    feature_cache = build_feature_cache(recordings, max_windows_per_recording)
    fold_results: list[dict] = []
    all_evaluated: list[dict] = []

    for fold_number, (train, test) in enumerate(
        cross_validation_splits(recordings, folds, seed), start=1
    ):
        train_artists = {recording.artist for recording in train}
        test_artists = {recording.artist for recording in test}
        if not train_artists.isdisjoint(test_artists):
            raise AssertionError("Artist leakage detected")

        print(
            f"fold {fold_number}/{folds}: fitting {len(train)} train / {len(test)} test",
            flush=True,
        )
        model = fit_classifier(train, feature_cache, classes, seed + fold_number)
        evaluated = evaluate_classifier(model, test, feature_cache, classes)
        top1 = sum(item["expected"] == item["predicted"] for item in evaluated)
        top3 = sum(item["expected"] in item["top3"] for item in evaluated)
        train_titles = {_normalized_title(recording) for recording in train}
        test_titles = {_normalized_title(recording) for recording in test}
        fold_dir = output_dir / f"fold-{fold_number}"
        save_model_bundle(model, classes, fold_dir)
        fold_result = {
            "fold": fold_number,
            "train_recordings": len(train),
            "test_recordings": len(evaluated),
            "train_artists": len(train_artists),
            "test_artists": len(test_artists),
            "top1_accuracy": top1 / len(evaluated),
            "top3_accuracy": top3 / len(evaluated),
            "exact_title_overlap": len(train_titles & test_titles),
            "evaluated": evaluated,
        }
        fold_results.append(fold_result)
        all_evaluated.extend(evaluated)
        print(
            f"fold {fold_number}: top-1={fold_result['top1_accuracy']:.3f}, "
            f"top-3={fold_result['top3_accuracy']:.3f}",
            flush=True,
        )

    confusion: dict[str, Counter] = defaultdict(Counter)
    for item in all_evaluated:
        confusion[item["expected"]][item["predicted"]] += 1
    total = len(all_evaluated)
    metrics = {
        "classes": classes,
        "folds": fold_results,
        "recordings": len(recordings),
        "artists": len({recording.artist for recording in recordings}),
        "feature_windows": sum(len(windows) for windows in feature_cache.values()),
        "aggregate_top1_accuracy": sum(
            item["expected"] == item["predicted"] for item in all_evaluated
        )
        / total,
        "aggregate_top3_accuracy": sum(
            item["expected"] in item["top3"] for item in all_evaluated
        )
        / total,
        "confusion": {raga: dict(counts) for raga, counts in sorted(confusion.items())},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    return metrics
