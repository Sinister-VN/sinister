import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text


def train_model(
    data_path: str = "data/FakeNewsNet.csv",
    model_dir: str = "saved_model",
) -> None:
    data = pd.read_csv(data_path)

    if "title" not in data.columns or "real" not in data.columns:
        raise ValueError("Dataset must include 'title' and 'real' columns.")

    data["title"] = data["title"].fillna("").astype(str)
    data["clean_text"] = data["title"].apply(clean_text)

    X = data["clean_text"]
    y = data["real"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
