from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import os
from app.preprocessing import preprocess  # Your custom cleaner
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI()

# CORS setup (for frontend/backend testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and responses.json from correct paths
# Correct path for model and responses
model_path = os.path.join("app", "models", "model.pkl")
responses_path = os.path.join("app","data","responses.json")

try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    print("✅ Responses loaded successfully")
except Exception as e:
    print(f"❌ Error loading responses: {e}")

# Schema for user input
class UserInput(BaseModel):
    text: str

# Global flag to show greeting once
greeted = False

# Language detection (very basic)
def detect_language(text):
    roman_keywords = ['kya', 'kaise', 'kr', 'nhi', 'banani', 'kaha', 'tamam', 'aaj', 'aj',
                      'karti', 'hain', 'mein', 'aur', 'karo', 'usme', 'par', 'ya', 'ko', 
                      'ke', 'jis', 'liye', 'nahi', 'kar', 'liye', 'liyae', 'par', 'sakte', 'naam', 'jao']
    if any(word in text.lower() for word in roman_keywords):
        return "roman"
    return "eng"

# Chat route
@app.post("/chat")
def chat(user_input: UserInput):
    global greeted
    cleaned_input = preprocess(user_input.text)
    print("🧹 Cleaned Input:", cleaned_input)

    # Detect language
    language = detect_language(user_input.text)
    print("🌐 Detected Language:", language)

    try:
        # Predict intent
        intent = model.predict([cleaned_input])[0]
        print("🔍 Predicted Intent:", intent)

        # Get response from JSON
        full_response = responses.get(intent.strip())

        if not full_response:
            raise Exception("Response not found for this intent")

        # Split response by \n\n
        if "\n\n" in full_response:
            eng_response, roman_response = full_response.split("\n\n", 1)
        else:
            eng_response, roman_response = full_response, full_response  # fallback

        # Choose response based on language
        reply = roman_response.strip() if language == "roman" else eng_response.strip()

        # First-time greeting
        if not greeted:
            greeted = True
            greeting = "Assalamualaikum! 👋 " if language == "roman" else "Hello! 👋 "
            return {"response": greeting + reply}

        return {"response": reply}

    except Exception as e:
        print(f"❌ Error: {e}")
        if language == "roman":
            return {"response": "Plz type again, kuch samajh nahi aya 😕"}
        else:
            return {"response": "Please type again, I didn't understand that 😕"}
