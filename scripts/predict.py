from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from news_classifier.config import settings
from news_classifier.predict import predict_text_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a label for one text string.")
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.default_model_artifact,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_text_from_path(args.text, args.model_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
