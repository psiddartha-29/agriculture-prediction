import logging
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("agri_api")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

YIELD_FILE = "best_yield_prediction_model.pkl"
FERTILIZER_FILE = "fertilizer_prediction_model1.pkl"
PESTICIDE_FILE = "pesticide_prediction_model.pkl"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def diagnose_model_file(path: Path) -> str | None:
    """Return a human-readable problem with the file, or None if it looks OK."""
    if not path.exists():
        return f"Model file not found: {path}"
    if path.stat().st_size == 0:
        return f"Model file is empty: {path.name}"

    with open(path, "rb") as f:
        head = f.read(64)

    if head.startswith(b"version https://git-lfs"):
        return (
            f"{path.name} is a Git LFS pointer, not the real model. "
            "Run 'git lfs pull' or re-download the actual file."
        )
    if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
        return f"{path.name} contains HTML (a failed download?), not a model. Re-download it."
    return None


@lru_cache(maxsize=1)  # only ONE model stays in RAM at a time
def get_model(name: str):
    path = MODELS_DIR / name
    problem = diagnose_model_file(path)
    if problem:
        raise RuntimeError(problem)
    return joblib.load(path)


def load_model_or_503(name: str):
    """Load a model; turn load failures into a clear 503 instead of a bare 500."""
    try:
        return get_model(name)
    except Exception as exc:
        logger.exception("Could not load model %s", name)
        raise HTTPException(
            status_code=503,
            detail=f"Model '{name}' could not be loaded: {exc}",
        )


def run_prediction(model_file: str, row: dict) -> float:
    """Build the DataFrame, align it to the model's training columns, and predict."""
    model = load_model_or_503(model_file)
    df = pd.DataFrame([row])

    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)

        # Tolerate Annual_Rainfall vs Annual_Rainfall_mm naming differences
        if "Annual_Rainfall" in expected and "Annual_Rainfall" not in df.columns:
            df = df.rename(columns={"Annual_Rainfall_mm": "Annual_Rainfall"})
        elif "Annual_Rainfall_mm" in expected and "Annual_Rainfall_mm" not in df.columns:
            df = df.rename(columns={"Annual_Rainfall": "Annual_Rainfall_mm"})

        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Model expects columns the API does not provide: {missing}. "
                    f"Model columns: {expected}"
                ),
            )
        df = df[expected]  # same columns, same order as training

    try:
        return float(model.predict(df)[0])
    except ValueError as exc:
        # e.g. unseen category like an unknown crop/state/season
        logger.exception("Prediction failed for %s", model_file)
        raise HTTPException(status_code=422, detail=f"Invalid input for model: {exc}")


# ---------------------------------------------------------------------------
# App + request schemas
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agricultural Yield Prediction",
    description="API for predicting agricultural crop yield",
    version="1.0",
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Agricultural Yield Prediction API is running"}


def file_info(name: str) -> dict:
    path = MODELS_DIR / name
    if not path.exists():
        return {"found": False, "ok": False, "problem": "file not found"}
    problem = diagnose_model_file(path)
    return {
        "found": True,
        "size_bytes": path.stat().st_size,
        "ok": problem is None,
        "problem": problem,
    }


@app.get("/health")
def health():
    # Inspects files without loading them (loading would use RAM)
    return {
        "status": "API is working",
        "yield_model": file_info(YIELD_FILE),
        "fertilizer_model": file_info(FERTILIZER_FILE),
        "pesticide_model": file_info(PESTICIDE_FILE),
    }


@app.post("/predict")
def predict_yield(data: YieldRequest):
    row = {
        "Crop": data.crop,
        "Crop_Year": data.crop_year,
        "Season": data.season,
        "State": data.state,
        "Area_acres": data.area_acres,
        "Annual_Rainfall_mm": data.annual_rainfall_mm,
        "Fertilizer_kg": data.fertilizer_kg,
        "Pesticide_kg": data.pesticide_kg,
        "Fertilizer_per_acre": data.fertilizer_kg / data.area_acres,
        "Pesticide_per_acre": data.pesticide_kg / data.area_acres,
    }
    prediction = run_prediction(YIELD_FILE, row)

    return {
        "crop": data.crop,
        "state": data.state,
        "year": data.crop_year,
        "predicted_yield_kg_per_acre": round(prediction, 2),
    }


@app.post("/predict/fertilizer")
def predict_fertilizer(data: FertilizerRequest):
    row = {
        "Crop": data.crop,
        "Crop_Year": data.crop_year,
        "Season": data.season,
        "State": data.state,
        "Area_acres": data.area_acres,
        "Annual_Rainfall": data.annual_rainfall,
    }
    prediction = run_prediction(FERTILIZER_FILE, row)

    return {
        "crop": data.crop,
        "state": data.state,
        "year": data.crop_year,
        "predicted_fertilizer_quantity": round(prediction, 2),
    }


@app.post("/predict/pesticide")
def predict_pesticide(data: PesticideRequest):
    row = {
        "Crop": data.crop,
        "Crop_Year": data.crop_year,
        "Season": data.season,
        "State": data.state,
        "Area_acres": data.area_acres,
        "Annual_Rainfall_mm": data.annual_rainfall_mm,
    }
    prediction = run_prediction(PESTICIDE_FILE, row)

    return {
        "crop": data.crop,
        "state": data.state,
        "year": data.crop_year,
        "predicted_pesticide_quantity": round(prediction, 2),
    }