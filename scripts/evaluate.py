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
from news_classifier.evaluate import evaluate_saved_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model artifact.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.default_model_artifact,
    )
    parser.add_argument("--test-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=settings.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_saved_model(
        model_path=args.model_path,
        test_sample=args.test_sample,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
