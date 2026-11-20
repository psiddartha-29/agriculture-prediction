from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (PS1/)
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(MODELS_DIR / "best_yield_prediction_model.pkl")
fertilizer_model = joblib.load(MODELS_DIR / "fertilizer_prediction_model1.pkl")
pesticide_model = joblib.load(MODELS_DIR / "pesticide_prediction_model.pkl")

app = FastAPI(
    title="Agricultural Yield Prediction",
    description="API for predicting agricultural crop yield",
    version="1.0"
)


class YieldRequest(BaseModel):
    crop: str = Field(..., min_length=2)
    crop_year: int = Field(..., ge=2000, le=2035)
    season: str
    state: str
    area_acres: float = Field(..., gt=0)
    annual_rainfall_mm: float = Field(..., ge=0)
    fertilizer_kg: float = Field(..., ge=0)
    pesticide_kg: float = Field(..., ge=0)


class FertilizerRequest(BaseModel):
    crop: str = Field(..., min_length=2)
    crop_year: int = Field(..., ge=1997, le=2035)
    season: str
    state: str
    area_acres: float = Field(..., gt=0)
    annual_rainfall: float = Field(..., ge=0)


class PesticideRequest(BaseModel):
    crop: str = Field(..., min_length=2)
    crop_year: int = Field(..., ge=1997, le=2035)
    season: str
    state: str
    area_acres: float = Field(..., gt=0)
    annual_rainfall_mm: float = Field(..., ge=0)


@app.get("/")
def home():
    return {
        "message": "Agricultural Yield Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "API is working",
        "yield_model_loaded": True,
        "fertilizer_model_loaded": True,
        "pesticide_model_loaded": True
    }


@app.post("/predict")
def predict_yield(data: YieldRequest):
    input_data = pd.DataFrame([{
        "Crop": data.crop,
        "Crop_Year": data.crop_year,
        "Season": data.season,
        "State": data.state,
        "Area_acres": data.area_acres,
        "Annual_Rainfall_mm": data.annual_rainfall_mm,
        "Fertilizer_kg": data.fertilizer_kg,
        "Pesticide_kg": data.pesticide_kg,
        "Fertilizer_per_acre": data.fertilizer_kg / data.area_acres,
        "Pesticide_per_acre": data.pesticide_kg / data.area_acres
    }])

    prediction = model.predict(input_data)[0]

    return {
        "crop": data.crop,
        "state": data.state,
        "year": data.crop_year,
        "predicted_yield_kg_per_acre": round(float(prediction), 2)
    }


@app.post("/predict/fertilizer")
def predict_fertilizer(data: FertilizerRequest):
    input_data = pd.DataFrame([{
        "Crop": data.crop,
        "Crop_Year": data.crop_year,
        "Season": data.season,
        "State": data.state,
        "Area_acres": data.area_acres,
        "Annual_Rainfall": data.annual_rainfall
    }])

    prediction = fertilizer_model.predict(input_data)[0]

    return {
        "crop": data.crop,
        "state": data.state,
        "year": data.crop_year,
        "predicted_fertilizer_quantity": round(float(prediction), 2)
    }


@app.post("/predict/pesticide")
def predict_pesticide(data: PesticideRequest):
    input_data = pd.DataFrame([{
        "Crop": data.crop,
        "Crop_Year": data.crop_year,
        "Season": data.season,
        "State": data.state,
        "Area_acres": data.area_acres,
        "Annual_Rainfall_mm": data.annual_rainfall_mm
    }])

    prediction = pesticide_model.predict(input_data)[0]

    return {
        "crop": data.crop,
        "state": data.state,
        "year": data.crop_year,
        "predicted_pesticide_quantity": round(float(prediction), 2)
    }
