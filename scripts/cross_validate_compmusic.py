#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from raga_detector.compmusic import discover_recordings
from raga_detector.evaluation import cross_validate


DEFAULT_DATA = Path("data/raw/compmusic/extracted/RagaDataset/Carnatic")
DEFAULT_OUTPUT = Path("artifacts/evaluations/compmusic-artist-cv-v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Artist-disjoint CompMusic cross-validation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=40)
    args = parser.parse_args()

    recordings = discover_recordings(args.data)
    counts = Counter(recording.raga for recording in recordings)
    supported = {raga for raga, count in counts.items() if count >= args.folds}
    recordings = [recording for recording in recordings if recording.raga in supported]
    metrics = cross_validate(
        recordings,
        args.output,
        folds=args.folds,
        max_windows_per_recording=args.max_windows,
    )
    print(
        f"\nAggregate top-1: {metrics['aggregate_top1_accuracy']:.3f}\n"
        f"Aggregate top-3: {metrics['aggregate_top3_accuracy']:.3f}\n"
        f"Metrics: {args.output / 'metrics.json'}"
    )


if __name__ == "__main__":
    main()
