# app/train_model.py

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import preprocess

def train():
    # Load and preprocess data
    df = pd.read_csv("app/data/intents.csv")  # path relative to root when run from root
    print("🔄 Preprocessing questions...", end=" ")

    # Added progress print (shows dots)
    df['qs'] = df['qs'].apply(lambda x: print(".", end="") or preprocess(x))

    print(" ✅ Done")

    X = df['qs']
    y = df['intents']

    # Define pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Train model
    print("🧠 Training model...")
    pipe.fit(X, y)

    # Save model
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(pipe, "app/models/model.pkl")
    print("\n✅ Model trained and saved successfully at app/models/model.pkl")

if __name__ == "__main__":
    print("👋 train_model.py started")  # TEMP: Check if this runs
    train()
