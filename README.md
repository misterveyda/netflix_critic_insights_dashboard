# Review Analysis Starter

This project provides starter scripts to analyze sentiment from Amazon/IMDb reviews using two approaches:

- Rule-based scoring with VADER (`src/vader_score.py`)
- Naive Bayes classification with Scikit-learn (`src/train_nb.py`)

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

2. Prepare a CSV with at least a text column and (for training) a label column. Default column names: `review` and `label`.

3. Run VADER scoring:

```bash
python src/vader_score.py --input data/reviews.csv --text-col review --out vader_out.csv
```

4. Train a Naive Bayes model:

```bash
python src/train_nb.py --input data/labeled_reviews.csv --text-col review --label-col label --model-out model.joblib
```

5. Run the API locally:

```bash
uvicorn src.api:app --reload --port 8000
```

6. Open the browser UI:

- Navigate to `http://localhost:8000`
- Enter review text and choose `Predict with VADER` or `Predict with Naive Bayes`

Docker

Build and run the API container:

```bash
docker build -t review-sentiment-api .
docker run -p 8000:8000 review-sentiment-api
```

Then open `http://localhost:8000` to use the UI.

Files

- [src/vader_score.py](src/vader_score.py)
- [src/train_nb.py](src/train_nb.py)
- [requirements.txt](requirements.txt)

Dataset

Use any Kaggle / SNAP review dataset. Do not commit large datasets to this repo.

Next steps

- Add a notebook for EDA
- Add model evaluation and cross-validation
- Add a simple web demo or API
# netflix_critic_insights_dashboard
to be update promise.
