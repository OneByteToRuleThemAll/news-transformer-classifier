from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from news_classifier.data import normalize_text, prepare_split_frame, sample_frame


class DataTests(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text(" a\n\nb\tc "), "a b c")

    def test_sample_frame_returns_requested_size(self) -> None:
        df = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 2]})
        sampled = sample_frame(df, sample_size=2, seed=42)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(list(sampled.columns), ["text", "label"])

    def test_prepare_split_frame_adds_expected_columns(self) -> None:
        df = pd.DataFrame({"text": ["Market rally"], "label": [2]})
        prepared = prepare_split_frame(df, "train")
        self.assertEqual(prepared.loc[0, "label_name"], "Business")
        self.assertEqual(prepared.loc[0, "split"], "train")
        self.assertEqual(prepared.loc[0, "char_count"], len("Market rally"))
        self.assertEqual(prepared.loc[0, "word_count"], 2)


if __name__ == "__main__":
    unittest.main()
