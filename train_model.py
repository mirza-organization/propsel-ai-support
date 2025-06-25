# app/train_model.py

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
from preprocessing import preprocess 

def train():
    # Load data
    df = pd.read_csv("app/data/intents.csv")
    df['qs'] = df['qs'].apply(preprocess)

    X = df['qs']
    y = df['intents']

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)

    os.makedirs("app/models", exist_ok=True)
    joblib.dump(pipe, "app/models/model.pkl")
    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train()
