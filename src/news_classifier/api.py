from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from news_classifier.predict import predict_text


app = FastAPI(title="News Classifier API")


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return predict_text(request.text)
