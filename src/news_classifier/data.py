from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd

from news_classifier.config import LABEL_NAMES, settings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sample_frame(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True).copy()
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True).copy()


def prepare_split_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    frame = df.copy()
    frame["text"] = frame["text"].map(normalize_text)
    frame["label_name"] = frame["label"].map(LABEL_NAMES)
    frame["split"] = split_name
    frame["char_count"] = frame["text"].str.len()
    frame["word_count"] = frame["text"].str.split().str.len()
    return frame


def load_ag_news(
    train_sample: int | None = None,
    test_sample: int | None = None,
    seed: int = settings.seed,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset

    dataset = load_dataset(settings.dataset_name)
    train_df = sample_frame(dataset["train"].to_pandas(), train_sample, seed)
    test_df = sample_frame(dataset["test"].to_pandas(), test_sample, seed + 1)
    return prepare_split_frame(train_df, "train"), prepare_split_frame(test_df, "test")
