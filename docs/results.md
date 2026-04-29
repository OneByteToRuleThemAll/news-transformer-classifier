# Results

## Dataset split information

- Confirmed from the notebook: the workflow loads `ag_news` train and test splits.
- Exact split counts in this refactor document: `TBD`

## Training configuration

- Word baseline: TF-IDF word n-grams with `LinearSVC`
- Character baseline: TF-IDF character n-grams with `LinearSVC`
- Transformer baseline: `textattack/distilbert-base-uncased-ag-news`
- Random seed: `42`
- Full hyperparameter details beyond the extracted source defaults: see [notebooks/original_assessment.ipynb](/C:/Users/matty/news-transformer-classifier/notebooks/original_assessment.ipynb)

## Metrics table

Recorded from `kaggle-outputs/model_summary.csv`:

| Model | Clean Accuracy | Clean Macro-F1 | Avg Perturbed Accuracy | Avg Perturbed Macro-F1 | Worst Perturbed Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_distilbert` | 0.9478947368 | 0.9479012037 | 0.9406286550 | 0.9406508945 | 0.9306032292 |
| `char_ngram_svm` | 0.9272368421 | 0.9271439399 | 0.9225146199 | 0.9223920338 | 0.9168609154 |
| `word_ngram_svm` | 0.9222368421 | 0.9220362284 | 0.9171052632 | 0.9169213001 | 0.9107899042 |

## Known limitations

- This refactor does not introduce new experiments; it packages the existing workflow and recorded outputs.
- Exact dataset row counts are left as `TBD` in this document because they were not explicitly documented in the original notebook narrative or existing docs.
- No benchmark claims are made beyond the outputs already present in the repository.
