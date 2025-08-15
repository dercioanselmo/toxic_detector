from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from src.inference.predict import predict

app = FastAPI()

class TextInput(BaseModel):
    text: str

logging.basicConfig(filename='api_logs.log', level=logging.INFO)

@app.post("/predict")
def predict_toxicity(input: TextInput):
    if not input.text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        result = predict(input.text)
        logging.info(f"Prediction for '{input.text}': {result}")
        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run with uvicorn src.api.app:app --reload