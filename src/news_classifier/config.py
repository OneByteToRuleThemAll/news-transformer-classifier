from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}
LABEL_ORDER = [LABEL_NAMES[idx] for idx in sorted(LABEL_NAMES)]
TRANSFORMER_MODEL_NAME = "textattack/distilbert-base-uncased-ag-news"
DATASET_NAME = "ag_news"


@dataclass(frozen=True)
class Settings:
    dataset_name: str = DATASET_NAME
    transformer_model_name: str = TRANSFORMER_MODEL_NAME
    seed: int = 42
    default_model_name: str = "char_ngram_svm"
    artifact_dir: Path = Path("artifacts")
    default_model_artifact: Path = Path("artifacts/model.joblib")


settings = Settings()
