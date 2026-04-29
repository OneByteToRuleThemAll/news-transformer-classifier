from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from news_classifier import predict


class DummyModel:
    def predict(self, texts: list[str]) -> list[int]:
        if texts != ["Breaking news"]:
            raise AssertionError(texts)
        return [2]


class PredictTests(unittest.TestCase):
    def test_predict_text_uses_cached_bundle(self) -> None:
        predict._load_default_bundle.cache_clear()
        with patch.object(
            predict,
            "_load_default_bundle",
            return_value={
                "model_name": "dummy_model",
                "label_names": {2: "Business"},
                "model": DummyModel(),
            },
        ):
            result = predict.predict_text("  Breaking   news  ")

        self.assertEqual(
            result,
            {
                "text": "Breaking news",
                "label_id": 2,
                "label": "Business",
                "model_name": "dummy_model",
            },
        )

    def test_predict_text_from_path_loads_bundle(self) -> None:
        model = type("Model", (), {"predict": lambda self, texts: [1]})()
        with patch.object(
            predict,
            "load_model_bundle",
            return_value={
                "model_name": "loaded:test.joblib",
                "label_names": {1: "Sports"},
                "model": model,
            },
        ):
            result = predict.predict_text_from_path(
                "Match report",
                Path("artifacts/test.joblib"),
            )

        self.assertEqual(result["label"], "Sports")
        self.assertEqual(result["model_name"], "loaded:test.joblib")


if __name__ == "__main__":
    unittest.main()
