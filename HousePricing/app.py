from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI(title="California House Price Prediction API")

# Load model and scaler
model = joblib.load("HousePricePredictor.pkl")
scaler = joblib.load("scaler.pkl")


# Input data schema (must match dataset features)
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# Root endpoint
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict_price(data: HouseFeatures):
    # Convert input data to numpy array
    input_data = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Return response
    return {
        "predicted_price": float(prediction[0])
    }
