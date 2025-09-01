from fastapi import FastAPI
from pydantic import BaseModel
import joblib, gzip
import numpy as np

# load model (compressed .pkl.gz)
with gzip.open("crop_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

# load encoders (normal .pkl)
label_encoders = joblib.load("encoders.pkl")

app = FastAPI(title="🌱 Crop Recommendation API")

# Input schema for validation
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running 🚀"}

@app.post("/predict")
def predict(data: CropInput):
    try:
        features = [
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall
        ]

        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]

        return {"recommended_crop": str(prediction)}

    except Exception as e:
        return {"error": str(e)}
