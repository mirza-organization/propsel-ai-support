# app/model.py

import joblib
import json
import os
from preprocessing import preprocess

# Load trained model once when app starts
model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
model = joblib.load(model_path)
print("✅ Model loaded")


# Load responses from responses.json
responses_path = os.path.join(os.path.dirname(__file__), "data", "responses.json")
with open(responses_path, "r", encoding="utf-8") as f:
    responses = json.load(f)

def get_response(user_text: str) -> str:
    # Preprocess user input text
    cleaned_text = preprocess(user_text)
    
    # Predict intent (returns list, take first element)
    intent = model.predict([cleaned_text])[0]

    # Return the response text for predicted intent
    return responses.get(intent, "Sorry, I did not understand that. Please try again.")
