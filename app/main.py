from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict

app = FastAPI(title="Fake News Detection API")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictResponse)
def predict_news(request: PredictRequest) -> PredictResponse:
    result = predict(request.text)
    return PredictResponse(prediction=result)
