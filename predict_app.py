import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os

# -------------------------------------------------------
# Load the trained model at startup
# -------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. "
        "Please run train.py first to generate model/model.pkl"
    )

# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",
    description="ML API serving House Price Estimator Model",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files if static directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Welcome to House Price Prediction API. Go to /docs for API documentation."}

# -------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------
class HouseFeatures(BaseModel):
    LotArea:     int   = Field(..., example=8450,  description="Lot size in square feet")
    OverallQual: int   = Field(..., example=7,     description="Overall material quality (1-10)")
    OverallCond: int   = Field(..., example=5,     description="Overall condition (1-10)")
    YearBuilt:   int   = Field(..., example=2003,  description="Year the house was built")
    GrLivArea:   int   = Field(..., example=1710,  description="Above ground living area in sq ft")
    GarageCars:  int   = Field(..., example=2,     description="Number of cars garage can hold")

class PredictionResponse(BaseModel):
    predicted_price: float
    model:           str
    status:          str

# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "House Price Estimator API is running!",
        "docs":    "/docs",
        "predict": "/predict"
    }


@app.get("/health", tags=["Health"])
def health():
    """Kubernetes readiness/liveness probe endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HouseFeatures):
    """
    Predict the sale price of a house.

    Send house features as JSON and receive the predicted SalePrice.
    """
    try:
        # Convert input to model-ready format
        input_data = np.array([[
            features.LotArea,
            features.OverallQual,
            features.OverallCond,
            features.YearBuilt,
            features.GrLivArea,
            features.GarageCars
        ]])

        predicted_price = model.predict(input_data)[0]

        return PredictionResponse(
            predicted_price=round(float(predicted_price), 2),
            model="RandomForestRegressor",
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
