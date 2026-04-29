from __future__ import annotations

import pandas as pd

from news_classifier.data import normalize_text, prepare_split_frame, sample_frame


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text(" a\n\nb\tc ") == "a b c"


def test_sample_frame_returns_requested_size() -> None:
    df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 2]})
    sampled = sample_frame(df, sample_size=2, seed=42)
    assert len(sampled) == 2
    assert list(sampled.columns) == ["text", "label"]


def test_prepare_split_frame_adds_expected_columns() -> None:
    df = pd.DataFrame({"text": ["Market rally"], "label": [2]})
    prepared = prepare_split_frame(df, "train")
    assert prepared.loc[0, "label_name"] == "Business"
    assert prepared.loc[0, "split"] == "train"
    assert prepared.loc[0, "char_count"] == len("Market rally")
    assert prepared.loc[0, "word_count"] == 2
