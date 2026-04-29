# News Transformer Classifier

This repository is an applied transformer NLP text-classification project refactored from a standalone notebook workflow into a reproducible Python package, CLI scripts, tests, and a small API.

The preserved source notebook lives at [notebooks/original_assessment.ipynb](/C:/Users/matty/news-transformer-classifier/notebooks/original_assessment.ipynb) and remains in the repository as provenance for the original assessment workflow.

## Confirmed project details

- Dataset: `ag_news`
- Base transformer model: `textattack/distilbert-base-uncased-ag-news`
- Labels: `World`, `Sports`, `Business`, `Sci/Tech`
- Current metrics: taken from the existing `kaggle-outputs/model_summary.csv`

| Model | Clean Accuracy | Clean Macro-F1 | Avg Perturbed Macro-F1 | Worst Perturbed Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `transformer_distilbert` | 0.9479 | 0.9479 | 0.9407 | 0.9306 |
| `char_ngram_svm` | 0.9272 | 0.9271 | 0.9224 | 0.9169 |
| `word_ngram_svm` | 0.9222 | 0.9220 | 0.9169 | 0.9108 |

If you regenerate results and they differ, treat the repository outputs as the currently recorded values and update the docs accordingly.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Project layout

- `src/news_classifier/`: reusable package code
- `scripts/`: CLI entry points for training, evaluation, and prediction
- `docs/model_card.md`: model card with confirmed details and `TBD` where needed
- `docs/results.md`: current recorded results and limitations
- `notebooks/original_assessment.ipynb`: preserved notebook provenance

## Commands

Train a model artifact:

```bash
python scripts/train.py --model-name char_ngram_svm --output-path artifacts/model.joblib
```

Evaluate a saved model artifact:

```bash
python scripts/evaluate.py --model-path artifacts/model.joblib
```

Run a single prediction from the CLI:

```bash
python scripts/predict.py --model-path artifacts/model.joblib --text "Stocks rise after upbeat earnings report"
```

Serve the FastAPI app:

```bash
uvicorn src.news_classifier.api:app --reload
```

Example prediction command:

```bash
python scripts/predict.py --model-path artifacts/model.joblib --text "The team secured a dramatic late victory"
```

## Notes

- The notebook has not been removed; it is preserved as provenance under `notebooks/original_assessment.ipynb`.
- No trained model binaries, checkpoints, or caches are committed as part of this refactor.
- Any fields not directly confirmed by the notebook or recorded outputs are marked as `TBD`.
