from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from news_classifier.config import LABEL_NAMES, settings
from news_classifier.data import normalize_text
from news_classifier.train import load_model_bundle


def _resolve_model_path() -> Path:
    override = os.getenv("NEWS_CLASSIFIER_MODEL_PATH")
    if override:
        return Path(override)
    return settings.default_model_artifact


@lru_cache(maxsize=1)
def _load_default_bundle() -> dict[str, Any]:
    model_path = _resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. Run scripts/train.py first "
            "or set NEWS_CLASSIFIER_MODEL_PATH to a saved artifact."
        )
    return load_model_bundle(model_path)


def _predict_with_bundle(text: str, bundle: dict[str, Any]) -> dict[str, Any]:
    cleaned = normalize_text(text)
    label_idx = int(bundle["model"].predict([cleaned])[0])
    labels = bundle.get("label_names", LABEL_NAMES)
    label_name = labels.get(label_idx, str(label_idx))
    return {
        "text": cleaned,
        "label_id": label_idx,
        "label": label_name,
        "model_name": bundle.get("model_name", "unknown"),
    }


def predict_text(text: str) -> dict[str, Any]:
    return _predict_with_bundle(text, _load_default_bundle())


def predict_text_from_path(text: str, model_path: Path) -> dict[str, Any]:
    bundle = load_model_bundle(model_path)
    return _predict_with_bundle(text, bundle)
