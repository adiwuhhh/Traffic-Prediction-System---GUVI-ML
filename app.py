"""
app.py — FastAPI server for traffic situation prediction
Run:   uvicorn app:app --reload
Docs:  http://localhost:8000/docs
"""

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.pkl")
DAY_ORDER  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEEKEND    = {"Saturday","Sunday"}

app = FastAPI(
    title="Traffic EDA — Prediction API",
    description="Predict urban traffic situation from vehicle counts and time features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ── Model loading ──────────────────────────────────────────────────────────────
_bundle = None

def get_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python src/train.py` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    hour:       int   = Field(..., ge=0, le=23,  description="Hour of day (0–23)")
    minute:     int   = Field(0,   ge=0, le=45,  description="Minute (0/15/30/45)")
    day:        Literal["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    car_count:  int   = Field(..., ge=0, description="Number of cars in 15-min window")
    bike_count: int   = Field(0,   ge=0)
    bus_count:  int   = Field(0,   ge=0)
    truck_count:int   = Field(0,   ge=0)

    model_config = {"json_schema_extra": {"example": {
        "hour": 8, "minute": 0, "day": "Monday",
        "car_count": 40, "bike_count": 5, "bus_count": 3, "truck_count": 10
    }}}


class PredictResponse(BaseModel):
    situation:     str
    probabilities: dict[str, float]
    total_vehicles:int


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    bundle = get_model()
    model  = bundle["model"]
    le     = bundle["label_encoder"]

    total = req.car_count + req.bike_count + req.bus_count + req.truck_count
    if total == 0:
        car_f = bike_f = bus_f = truck_f = 0.0
    else:
        car_f   = req.car_count   / total
        bike_f  = req.bike_count  / total
        bus_f   = req.bus_count   / total
        truck_f = req.truck_count / total

    is_weekend  = int(req.day in WEEKEND)
    day_ordinal = DAY_ORDER.index(req.day)

    X = np.array([[
        req.hour, req.minute, is_weekend,
        req.car_count, req.bike_count, req.bus_count, req.truck_count,
        car_f, bike_f, bus_f, truck_f,
        day_ordinal,
    ]])

    y_enc = model.predict(X)[0]
    label = le.inverse_transform([y_enc])[0]

    proba = {}
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[0]
        proba = {cls: round(float(p), 4) for cls, p in zip(le.classes_, raw)}
    else:
        proba = {label: 1.0}

    return PredictResponse(
        situation=label,
        probabilities=proba,
        total_vehicles=total,
    )
