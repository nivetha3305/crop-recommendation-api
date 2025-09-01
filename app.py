from fastapi import FastAPI
from pydantic import BaseModel
import joblib, gzip
import numpy as np

# Load the compressed model
with gzip.open("crop_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

# Load encoders (normal .pkl)
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
    soil_type: str
    season: str
    region: str

@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running 🚀"}

@app.post("/predict")
def predict(data: CropInput):
    try:
        # encode categorical values using encoders.pkl
        st_enc = label_encoders["soil_type"].transform([data.soil_type])[0]
        se_enc = label_encoders["season"].transform([data.season])[0]
        rg_enc = label_encoders["region"].transform([data.region])[0]

        # final feature list (10 features)
        features = [
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall,
            st_enc,
            se_enc,
            rg_enc
        ]

        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]

        # decode crop label back to name
        crop_name = label_encoders["label"].inverse_transform([prediction])[0]

        return {"recommended_crop": crop_name}

    except Exception as e:
        return {"error": str(e)}
