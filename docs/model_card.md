# Model Card

## Model name

`transformer_distilbert`

## Base model

`textattack/distilbert-base-uncased-ag-news`

## Dataset

`ag_news`

## Task

Multi-class news text classification.

## Labels

- `World`
- `Sports`
- `Business`
- `Sci/Tech`

## Intended use

This project is intended as an applied NLP portfolio project and assessment refactor. The package, scripts, and API are designed to demonstrate reproducible text-classification workflows derived from the preserved notebook.

## Limitations

- The repository does not commit trained model artifacts.
- Reproducing transformer inference may require a compatible local environment and model download.
- The notebook benchmark includes robustness experiments, but this refactor does not claim any new benchmark beyond the recorded outputs already present in the repository.

## Evaluation metrics

- Clean accuracy: `0.9479`
- Clean macro-F1: `0.9479`
- Average perturbed macro-F1: `0.9407`
- Worst perturbed macro-F1: `0.9306`

## Ethical considerations

- News classification systems can encode source, topic, or language biases from their training data.
- Predictions should not be treated as factual verification.
- Out-of-domain or short text inputs may behave unpredictably.
