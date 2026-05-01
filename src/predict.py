import os
import pickle

from src.preprocessing import clean_text


def _load_artifacts(model_dir: str = "saved_model"):
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer not found. Train the model first.")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, vectorizer


def predict(text: str) -> str:
    model, vectorizer = _load_artifacts()
    cleaned = clean_text(text or "")
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]
    return "Real News" if int(label) == 1 else "Fake News"
