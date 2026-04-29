from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from news_classifier.config import settings
from news_classifier.train import save_model_bundle, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a saved model artifact.")
    parser.add_argument(
        "--model-name",
        default=settings.default_model_name,
        choices=["word_ngram_svm", "char_ngram_svm", "transformer_distilbert"],
    )
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=settings.default_model_artifact,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = train_model(
        model_name=args.model_name,
        train_sample=args.train_sample,
        seed=args.seed,
    )
    output_path = save_model_bundle(bundle, args.output_path)
    print(output_path)


if __name__ == "__main__":
    main()
