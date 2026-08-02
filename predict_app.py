import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="House Price Estimator API",
    description="Production MLOps API for House Price Prediction",
    version="1.0.0"
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
    LotArea: float = Field(..., gt=0, description="Lot size in square feet", example=8500.0)
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish rating (1-10)", example=7)
    OverallCond: int = Field(..., ge=1, le=10, description="Overall condition rating (1-10)", example=5)
    YearBuilt: int = Field(..., ge=1800, le=2026, description="Original construction date", example=2008)
    GrLivArea: float = Field(..., gt=0, description="Above grade (ground) living area square feet", example=1800.0)
    GarageCars: int = Field(..., ge=0, le=10, description="Size of garage in car capacity", example=2)

class PredictionOutput(BaseModel):
    predicted_price: float = Field(..., description="Estimated house price in INR (₹)")
    price_lakhs: float = Field(..., description="Estimated price in Lakhs (₹)")
    currency: str = "INR (₹)"
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
        "model_type": "RandomForestRegressor Pipeline",
        "features": ["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea", "GarageCars"],
        "target": "SalePrice (₹)"
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_price(input_data: HouseInput):
    try:
        pipeline = get_model()

        df_input = pd.DataFrame([input_data.model_dump()])

        prediction = pipeline.predict(df_input)[0]

        price_lakhs = round(prediction / 100000.0, 2)

        return PredictionOutput(
            predicted_price=round(float(prediction), 2),
            price_lakhs=price_lakhs
        )

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
