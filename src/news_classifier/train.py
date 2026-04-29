from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from news_classifier.config import LABEL_NAMES, TRANSFORMER_MODEL_NAME
from news_classifier.data import load_ag_news


def build_word_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=60000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(C=1.5)),
        ]
    )


def build_char_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=90000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(C=1.0)),
        ]
    )


class PretrainedTransformerClassifier:
    """Lightweight wrapper around the notebook's pretrained classifier path."""

    def __init__(
        self,
        model_name: str = TRANSFORMER_MODEL_NAME,
        batch_size: int = 32,
        max_length: int = 128,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def fit(self, X: list[str], y: list[int]) -> "PretrainedTransformerClassifier":
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        predictions: list[int] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                logits = self.model(**encoded).logits
            predictions.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
        return np.asarray(predictions, dtype=int)


def build_transformer_model(
    model_name: str = TRANSFORMER_MODEL_NAME,
    batch_size: int = 32,
    max_length: int = 128,
) -> PretrainedTransformerClassifier:
    return PretrainedTransformerClassifier(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )


def build_model(model_name: str) -> Any:
    if model_name == "word_ngram_svm":
        return build_word_model()
    if model_name == "char_ngram_svm":
        return build_char_model()
    if model_name == "transformer_distilbert":
        return build_transformer_model()
    raise ValueError(f"Unsupported model_name: {model_name}")


def train_model(
    model_name: str,
    train_sample: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    train_df, _ = load_ag_news(train_sample=train_sample, test_sample=None, seed=seed)
    model = build_model(model_name)
    model.fit(train_df["text"].tolist(), train_df["label"].tolist())
    return {
        "model_name": model_name,
        "dataset_name": "ag_news",
        "label_names": LABEL_NAMES,
        "model": model,
    }


def save_model_bundle(bundle: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return output_path


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)
