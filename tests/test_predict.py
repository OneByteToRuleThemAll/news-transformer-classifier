from __future__ import annotations

from pathlib import Path

from news_classifier import predict


class DummyModel:
    def predict(self, texts: list[str]) -> list[int]:
        assert texts == ["Breaking news"]
        return [2]


def test_predict_text_uses_cached_bundle(monkeypatch) -> None:
    predict._load_default_bundle.cache_clear()
    monkeypatch.setattr(
        predict,
        "_load_default_bundle",
        lambda: {
            "model_name": "dummy_model",
            "label_names": {2: "Business"},
            "model": DummyModel(),
        },
    )
    result = predict.predict_text("  Breaking   news  ")
    assert result == {
        "text": "Breaking news",
        "label_id": 2,
        "label": "Business",
        "model_name": "dummy_model",
    }


def test_predict_text_from_path_loads_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        predict,
        "load_model_bundle",
        lambda path: {
            "model_name": f"loaded:{Path(path).name}",
            "label_names": {1: "Sports"},
            "model": type("Model", (), {"predict": lambda self, texts: [1]})(),
        },
    )
    result = predict.predict_text_from_path("Match report", Path("artifacts/test.joblib"))
    assert result["label"] == "Sports"
    assert result["model_name"] == "loaded:test.joblib"
