#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from raga_detector.compmusic import TARGET_ALIASES, discover_recordings
from raga_detector.training import split_by_artist, train_model


DEFAULT_DATA = Path("data/raw/compmusic/extracted/RagaDataset/Carnatic")
DEFAULT_OUTPUT = Path("artifacts/models/compmusic-target-v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train our raga model on CompMusic pitch data")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-windows", type=int, default=40)
    args = parser.parse_args()

    recordings = discover_recordings(args.data)
    counts = Counter(recording.raga for recording in recordings)
    print("Dataset coverage:")
    for raga in TARGET_ALIASES:
        print(f"  {raga:20} recordings={counts[raga]:2}")

    supported = {raga for raga, count in counts.items() if count >= 2}
    usable = [recording for recording in recordings if recording.raga in supported]
    train, test = split_by_artist(usable)
    print(
        f"\nArtist-disjoint split: {len(train)} train recordings / "
        f"{len(test)} test recordings\n"
    )
    metrics = train_model(train, test, args.output, args.max_windows)
    print(
        f"\nModel saved to {args.output}\n"
        f"Recording-level top-1: {metrics['top1_accuracy']:.3f}\n"
        f"Recording-level top-3: {metrics['top3_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
