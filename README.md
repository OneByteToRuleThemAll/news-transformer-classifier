# Assessment 2 Applied NLP Project

## How to Run

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Open [`Assessment-2-AG-NEWS.ipynb`](Assessment-2-AG-NEWS.ipynb) in Jupyter or VS Code.

3. Run the notebook from top to bottom.

4. If you want to reproduce the transformer results quickly, use a GPU environment. The notebook was run on Kaggle with a T4 GPU. Total runtime was about 7 minutes.

## Notes

- The project uses the AG News dataset from Hugging Face.
- The pretrained transformer checkpoint is `textattack/distilbert-base-uncased-ag-news`.
- Final figures and metrics are saved in [`kaggle-outputs/`](kaggle-outputs/).
