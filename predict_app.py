import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="House Price Estimator API",
    description="Production MLOps API for Ames Housing Price Prediction",
    version="2.0.0"
)

# Serve static HTML/JS frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "model/model.pkl"
model_pipeline = None

def get_model():
    global model_pipeline
    if model_pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        model_pipeline = joblib.load(MODEL_PATH)
    return model_pipeline

class HouseInput(BaseModel):
    LotArea: float = Field(8450.0, gt=0, description="Lot size in square feet")
    OverallQual: int = Field(7, ge=1, le=10, description="Overall material and finish rating (1-10)")
    OverallCond: int = Field(5, ge=1, le=10, description="Overall condition rating (1-10)")
    YearBuilt: int = Field(2003, ge=1800, le=2026, description="Original construction date")
    YearRemodAdd: int = Field(2003, ge=1800, le=2026, description="Remodel date")
    TotalBsmtSF: float = Field(856.0, ge=0, description="Total basement area in sq ft")
    FirstFlrSF: float = Field(856.0, gt=0, alias="1stFlrSF", description="First floor living area in sq ft")
    SecondFlrSF: float = Field(854.0, ge=0, alias="2ndFlrSF", description="Second floor living area in sq ft")
    GrLivArea: float = Field(1710.0, gt=0, description="Above grade living area in sq ft")
    FullBath: int = Field(2, ge=0, le=10, description="Full bathrooms above grade")
    HalfBath: int = Field(1, ge=0, le=10, description="Half baths above grade")
    BedroomAbvGr: int = Field(3, ge=0, le=10, description="Bedrooms above grade")
    TotRmsAbvGrd: int = Field(8, ge=1, le=20, description="Total rooms above grade")
    GarageCars: int = Field(2, ge=0, le=10, description="Size of garage in car capacity")
    GarageArea: float = Field(548.0, ge=0, description="Garage size in sq ft")

    class Config:
        populate_by_name = True

class PredictionOutput(BaseModel):
    predicted_price: float = Field(..., description="Estimated house price in USD ($)")
    predicted_price_inr: float = Field(..., description="Estimated house price in INR (₹)")
    price_lakhs: float = Field(..., description="Estimated price in Lakhs (₹)")
    currency: str = "USD ($) / INR (₹)"
    status: str = "success"

@app.get("/")
def serve_ui():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={"message": "House Price Prediction API is active. Visit /docs for API documentation."})

@app.get("/health")
def health_check():
    try:
        model = get_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

@app.get("/info")
def model_info():
    return {
        "model_type": "RandomForestRegressor Pipeline (Ames Housing Dataset)",
        "features": [
            "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
            "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath",
            "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "GarageCars", "GarageArea"
        ],
        "target": "SalePrice ($ / ₹)"
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_price(input_data: HouseInput):
    try:
        pipeline = get_model()

        # Map field names using aliases ('1stFlrSF', '2ndFlrSF')
        df_input = pd.DataFrame([input_data.model_dump(by_alias=True)])

        prediction_usd = pipeline.predict(df_input)[0]
        prediction_inr = prediction_usd * 83.0  # 1 USD ≈ 83 INR
        price_lakhs = round(prediction_inr / 100000.0, 2)

        return PredictionOutput(
            predicted_price=round(float(prediction_usd), 2),
            predicted_price_inr=round(float(prediction_inr), 2),
            price_lakhs=price_lakhs
        )

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
