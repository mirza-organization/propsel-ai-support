# app/preprocessing.py

import re

def preprocess(text: str) -> str:
    """
    Lowercases and removes punctuation/extra spaces.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    return text
