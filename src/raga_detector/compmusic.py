from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


@dataclass(frozen=True)
class PitchRecording:
    track_id: str
    raga: str
    artist: str
    pitch_path: Path
    tonic_path: Path


def _plain(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if character.isascii()).lower().replace(" ", "")


def discover_recordings(carnatic_root: Path) -> list[PitchRecording]:
    mapping_path = carnatic_root / "_info_" / "ragaId_to_ragaName_mapping.json"
    feature_root = carnatic_root / "features"
    mapping = json.loads(mapping_path.read_text())
    normalized_to_target = {
        alias: target for target, aliases in TARGET_ALIASES.items() for alias in aliases
    }

    recordings: list[PitchRecording] = []
    for raga_id, source_name in mapping.items():
        target = normalized_to_target.get(_plain(source_name))
        raga_root = feature_root / raga_id
        if target is None or not raga_root.is_dir():
            continue
        for pitch_path in sorted(raga_root.rglob("*.pitch")):
            tonic_path = pitch_path.with_suffix(".tonic")
            if not tonic_path.is_file():
                continue
            relative = pitch_path.relative_to(raga_root)
            recordings.append(
                PitchRecording(
                    track_id=str(pitch_path.relative_to(feature_root).with_suffix("")),
                    raga=target,
                    artist=relative.parts[0],
                    pitch_path=pitch_path,
                    tonic_path=tonic_path,
                )
            )
    return recordings


def load_pitch(recording: PitchRecording) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.loadtxt(recording.pitch_path, dtype=np.float64, ndmin=2)
    if values.shape[1] < 2:
        raise ValueError(f"Pitch file needs time and frequency columns: {recording.pitch_path}")
    tonic_hz = float(recording.tonic_path.read_text().strip())
    return values[:, 0], values[:, 1], tonic_hz
