from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from news_classifier.data import load_ag_news
from news_classifier.train import load_model_bundle


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def evaluate_saved_model(
    model_path: Path,
    test_sample: int | None = None,
    seed: int = 42,
) -> dict[str, float | str | int]:
    bundle = load_model_bundle(model_path)
    _, test_df = load_ag_news(train_sample=None, test_sample=test_sample, seed=seed)
    y_true = test_df["label"].to_numpy()
    y_pred = bundle["model"].predict(test_df["text"].tolist())
    metrics = evaluate_predictions(y_true, y_pred)
    return {
        "model_name": str(bundle["model_name"]),
        "test_rows": int(len(test_df)),
        **metrics,
    }


def load_recorded_results(results_path: Path) -> pd.DataFrame:
    return pd.read_csv(results_path)
