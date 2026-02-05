import os
from fastapi import FastAPI, Header
from audio_utils import base64_to_wav
from predict import predict

API_KEY = os.getenv("API_KEY")  # default mat rakho

LANGS = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/api/voice-detection")
def detect(payload: dict, x_api_key: str = Header(None)):

    if not API_KEY:
        return {"status": "error", "message": "Server API key not set"}

    if x_api_key != API_KEY:
        return {"status": "error", "message": "Invalid API key"}

    try:
        if payload["language"] not in LANGS:
            raise ValueError("Invalid language")

        if payload["audioFormat"].lower() != "mp3":
            raise ValueError("Invalid audio format")

        audio = base64_to_wav(payload["audioBase64"])
        result = predict(audio)

        explanation = (
            "Synthetic speech patterns detected"
            if result["label"] == "AI_GENERATED"
            else "Natural speech characteristics detected"
        )

        return {
            "status": "success",
            "language": payload["language"],
            "classification": result["label"],
            "confidenceScore": result["confidence"],
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
