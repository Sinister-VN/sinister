# Fake News Detection App

A minimal, portfolio-ready fake news detector using classic NLP and a simple API.

## Problem
Given a news title, classify it as fake or real.

## Tech stack
- Python
- pandas, numpy, scikit-learn
- FastAPI + Uvicorn

## Project structure
- [data/](data): dataset CSV
- [notebooks/](notebooks): one clean EDA/training notebook
- [src/preprocessing.py](src/preprocessing.py): text cleaning
- [src/model.py](src/model.py): training and evaluation
- [src/predict.py](src/predict.py): inference pipeline
- [app/main.py](app/main.py): FastAPI app
- [saved_model/](saved_model): saved model artifacts

## Setup
Install dependencies:
```bash
py -m pip install -r requirements.txt
```

Train the model and save artifacts:
```bash
py -m src.model
```

Start the API:
```bash
py -m uvicorn app.main:app --reload
```

## API usage
Request:
```bash
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "The government announced a new policy today."}'
```

Response:
```json
{
  "prediction": "Real News"
}
```