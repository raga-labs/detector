#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path


TARGET_ALIASES = {
    "mayamalavagowla": {"mayamalavagowla", "mayamalavagaula"},
    "kharaharapriya": {"kharaharapriya", "karaharapriya"},
    "kalyani": {"kalyani"},
    "hindolam": {"hindolam"},
    "mohanam": {"mohanam"},
    "shankarabharanam": {"shankarabharanam", "sankarabharanam"},
    "bhairavi": {"bhairavi"},
    "mukhari": {"mukhari"},
    "saveri": {"saveri"},
    "kambhoji": {"kambhoji"},
    "dharmavati": {"dharmavati"},
    "todi": {"todi", "thodi"},
}


def plain(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if character.isascii()).lower()


def main() -> int:
    saraga_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/saraga-metadata")
    carnatic = saraga_root / "dataset" / "carnatic"
    if not carnatic.is_dir():
        print(f"Carnatic metadata directory not found: {carnatic}", file=sys.stderr)
        return 1

    counts: Counter[str] = Counter()
    total_recordings = 0
    labelled_recordings = 0
    for metadata_path in carnatic.rglob("*.json"):
        total_recordings += 1
        metadata = json.loads(metadata_path.read_text())
        ragas = metadata.get("raaga", [])
        if ragas:
            labelled_recordings += 1
        for raga in ragas:
            normalized = plain(raga.get("name", ""))
            for target, aliases in TARGET_ALIASES.items():
                if normalized in aliases:
                    counts[target] += 1

    print(f"Carnatic recordings: {total_recordings}")
    print(f"Recordings with a raga label: {labelled_recordings}")
    print("\nTarget-raga coverage:")
    for target in TARGET_ALIASES:
        print(f"{target:20} {counts[target]:3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
