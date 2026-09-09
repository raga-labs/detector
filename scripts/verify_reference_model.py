#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import xgboost as xgb


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "artifacts" / "reference" / "twelveswaras"


def main() -> None:
    model_path = MODEL_DIR / "raaga_xgb.json"
    classes_path = MODEL_DIR / "raaga_xgb.classes.json"
    mapping_path = ROOT / "config" / "reference-label-map.json"

    booster = xgb.Booster()
    booster.load_model(model_path)
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    missing = {
        raga_id: label
        for raga_id, label in mapping.items()
        if label is not None and label not in classes
    }
    if missing:
        raise RuntimeError(f"Mapped labels absent from reference model: {missing}")

    print(f"classes={len(classes)}")
    print(f"features={booster.num_features()}")
    print(f"boosted_rounds={booster.num_boosted_rounds()}")
    print(f"target_ragas_supported={sum(label is not None for label in mapping.values())}")
    print(f"target_ragas_pending={sum(label is None for label in mapping.values())}")
    print("reference_model=ok")


if __name__ == "__main__":
    main()
