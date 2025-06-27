# app/preprocessing.py

import re
from textblob import TextBlob  # Add this for typo correction

def preprocess(text: str) -> str:
    """
    Lowercases, corrects typos, and removes punctuation/extra spaces.
    """

    text = text.lower()  # Convert to lowercase
    text = str(TextBlob(text).correct())  # Correct common typos like 'dfasboard' → 'dashboard'
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
    return text
