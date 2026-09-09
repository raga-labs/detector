from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify a Carnatic raga from audio")
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/reference/twelveswaras"),
    )
    args = parser.parse_args()
    try:
        result = predict(args.audio, args.model_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
