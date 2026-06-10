from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.preprocess import clean_text

app = FastAPI(title="Review Sentiment API")
app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True),
    name="static",
)

analyzer = SentimentIntensityAnalyzer()
MODEL_PATH = "model.joblib"


class TextIn(BaseModel):
    text: str


@app.post("/predict/vader")
def predict_vader(payload: TextIn):
    t = clean_text(payload.text)
    return analyzer.polarity_scores(t)


@app.post("/predict/nb")
def predict_nb(payload: TextIn):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail=f"Model not found at {MODEL_PATH}. Train and save model first.")
    data = joblib.load(MODEL_PATH)
    vect = data["vectorizer"]
    model = data["model"]
    t = clean_text(payload.text)
    X = vect.transform([t])
    pred = model.predict(X)[0]
    return {"label": str(pred)}
